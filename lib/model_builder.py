from collections import namedtuple
import joblib
import numpy as np
import pyomo.environ as pyo
from collections import defaultdict
import pandas as pd

tyre_tuple = namedtuple("Tyre", ["id", "compound", "init_age"])

class F1StrategyOptimizer:
    def __init__(
        self,
        track: str,
        race_laps: int,
        available_tyres: tyre_tuple,
        pit_stop_overhead = 25.0,
        models_path = "./models",
        baseline_filename = "baselines.joblib",
        poly_filename = "poly_transform.joblib",
        tyre_models_filename = "tyre_models.joblib",
        max_age_map_filename = "max_age_map.joblib"
    ):
        self.track = track
        self.race_laps = race_laps

        if len(available_tyres) not in (6, 7):
            raise ValueError("Available tyres for race day must be 6 or 7")

        self.available_tyres = available_tyres
        self.pit_stop_overhead = pit_stop_overhead

        self.baselines_dict = joblib.load(models_path + "/" + baseline_filename)
        self.poly = joblib.load(models_path + "/" + poly_filename)
        self.models = joblib.load(models_path + "/" + tyre_models_filename)
        self.max_age_map = joblib.load(models_path + "/" + max_age_map_filename)

        self.avg_baselines_dict = self.create_avg_baseline()
        self.lap_time_data = self.precompute_stint_times()

    def create_avg_baseline(self):
        avg_baselines_dict = {}

        for (trk, comp, yr), baseline in self.baselines_dict.items():
            if trk != self.track:
                continue
            key = (trk, comp)
            if key not in avg_baselines_dict:
                avg_baselines_dict[key] = []
            avg_baselines_dict[key].append(baseline)

        for key in avg_baselines_dict:
            avg_baselines_dict[key] = sum(avg_baselines_dict[key]) / len(avg_baselines_dict[key])

        return avg_baselines_dict
    
    def predict_lap_time_for_age(self, track_name, compound, tyre_age):
        """
        Uses models (trained on lap_delta) and baselines_dict to reconstruct absolute lap time.
        Returns lap_duration in seconds (float).
        """
        # Check if model exists for (track, compound)
        key = (track_name, compound)
        if key not in self.models:
            raise KeyError(f"No model for {key}. Train per-track-per-compound models first.")

        model = self.models[key]
        X_poly = self.poly.transform(np.array([[tyre_age]]))
        lap_delta = float(model.predict(X_poly)[0])  # predicted delta vs year baseline
        baseline = self.avg_baselines_dict.get((track_name, compound))

        if baseline is None:
            raise KeyError(f"No baseline for {(track_name, compound)}. Cannot predict absolute lap time.")

        return baseline + lap_delta
    
    def precompute_stint_times(self):
        """
        Precompute stint times for all combinations.
        """
        lap_time_data = {}

        for tyre in self.available_tyres:
            tyre_id = tyre.id
            base_age = tyre.init_age
            max_age = self.max_age_map[(self.track, tyre.compound)]

            for l in range(1, min(self.race_laps, max_age) + 1):
                lap_time_data[(tyre_id, l)] = self.predict_lap_time_for_age(
                    track_name=self.track,
                    compound=tyre.compound,
                    tyre_age=base_age + l - 1
                )
        
        return lap_time_data

    def build_model(self):
        tyres_by_compound = defaultdict(list)

        for tyre in self.available_tyres:
            tyres_by_compound[tyre.compound].append(tyre.id)

        tyre_ids = [tyre.id for tyre in self.available_tyres]

        model = pyo.ConcreteModel()

        # -------------------------------------------------------------------------
        # SETS
        # -------------------------------------------------------------------------
        model.tyres = pyo.Set(initialize=tyre_ids) # Tyres available
        model.L = pyo.RangeSet(1, self.race_laps)
        model.compounds = pyo.Set(initialize=list(tyres_by_compound.keys()), ordered=False)
        model.tyres_by_compound = pyo.Set(
            model.compounds,
            initialize=lambda m, c: tyres_by_compound[c]
        )

        # -------------------------------------------------------------------------
        # PARAMETERS
        # -------------------------------------------------------------------------
        model.tyre_age = pyo.Param(model.tyres, initialize={tyre.id: tyre.init_age for tyre in self.available_tyres})
        model.tyre_age_max = pyo.Param(model.tyres, initialize={tyre.id: self.max_age_map[(self.track, tyre.compound)] for tyre in self.available_tyres})
        model.lap_time = pyo.Param(
            model.tyres,
            model.L,
            initialize=self.lap_time_data,
            domain=pyo.NonNegativeReals,
            default=0.0
        ) 

        # -------------------------------------------------------------------------
        # VARIABLES
        # -------------------------------------------------------------------------
        model.run_lap = pyo.Var(model.tyres, model.L, domain=pyo.Binary)
        model.use_stint = pyo.Var(model.tyres, domain=pyo.Binary) # Stint
        model.use_compound = pyo.Var(model.compounds, domain=pyo.Binary) # For mandatory min 2 types of compound used

        # -------------------------------------------------------------------------
        # CONSTRAINTS
        # -------------------------------------------------------------------------
        # Total race distance
        def total_laps_rule(m):
            return sum(m.run_lap[t, l] for t in m.tyres for l in m.L) == self.race_laps

        model.total_laps = pyo.Constraint(rule=total_laps_rule)

        # Can only run laps if tyre is used
        def link_stint_rule(m, t):
            return sum(m.run_lap[t, l] for l in m.L) <= self.race_laps * m.use_stint[t]

        model.link_stint = pyo.Constraint(model.tyres, rule=link_stint_rule)

        # Tyre age limit
        def age_limit_rule(m, t):
            return sum(m.run_lap[t, l] for l in m.L) <= (m.tyre_age_max[t] - m.tyre_age[t]) * m.use_stint[t]

        model.age_limit = pyo.Constraint(model.tyres, rule=age_limit_rule)

        # Contiguous laps (no skipping laps, must be sequential)
        def contiguity_rule(m, t, l):
            if l == self.race_laps:
                return pyo.Constraint.Skip
            return m.run_lap[t, l] >= m.run_lap[t, l + 1]

        model.contiguity = pyo.Constraint(model.tyres, model.L, rule=contiguity_rule)

        # If any stint of that compound is used then compound is used.
        def compound_usage_rule(m, c):
            return sum(
                sum(m.run_lap[t, l] for l in model.L) for t in m.tyres_by_compound[c]
            ) <= self.race_laps * m.use_compound[c]

        model.compound_usage = pyo.Constraint(
            model.compounds,
            rule=compound_usage_rule
        )

        # # If no stint of that compound is used then compound is not used.
        def compound_usage_rule2(m, c):
            return sum(
                sum(m.run_lap[t, l] for l in model.L) for t in m.tyres_by_compound[c]
            ) >= m.use_compound[c]

        model.compound_usage2 = pyo.Constraint(
            model.compounds,
            rule=compound_usage_rule2
        )

        # Mandatory compound rule; At least 2 different compounds used
        model.mandatory_compounds = pyo.Constraint(
            expr=sum(model.use_compound[c] for c in model.compounds) >= 2
        )

        MIN_LAPS_PER_COMPOUND = 1   # FIA minimum = 1

        def compound_min_laps_rule(m, c):
            return sum(sum(m.run_lap[t, l] for l in model.L) for t in m.tyres_by_compound[c]) >= MIN_LAPS_PER_COMPOUND * m.use_compound[c]

        model.compound_min_laps = pyo.Constraint(
            model.compounds,
            rule=compound_min_laps_rule
        )

        # -------------------------------------------------------------------------
        # OBJECTIVE: minimize total predicted race time
        # -------------------------------------------------------------------------
        def total_race_time(m):
            return (
                sum(
                    m.lap_time[tyre, lap] * m.run_lap[tyre, lap]
                    for tyre in m.tyres
                    for lap in m.L
                )
                + self.pit_stop_overhead * (sum(m.use_stint[tyre] for tyre in m.tyres) - 1)
            )

        model.obj = pyo.Objective(rule=total_race_time, sense=pyo.minimize)

        return model

    def get_tyre_info(self, tyre_id):
        return next((t for t in self.available_tyres if t.id == tyre_id), None)

    def print_results(self, results, model):
        print(f"Track        : {self.track}")
        print(f"Race Laps    : {self.race_laps}")
        print(f"Pit Stop Time: {self.pit_stop_overhead}")

        print("\nAvailable Tyres:")
        for t in self.available_tyres:
            print(f"  {t.id} {f'(Age: {t.init_age})' if t.init_age > 0 else '(Fresh)'}")

        print("\nMax Tyre Age:")
        for compound in ["SOFT", "MEDIUM", "HARD"]:
            print(f"  {compound}: {self.max_age_map[(self.track, compound)]}")

        if results.solver.termination_condition == "optimal":
            print(f"\n===== OPTIMIZED STRATEGY =====")

            total_time = pyo.value(model.obj)
            print(f"Total race time: {total_time:.2f} s")

            used_tyres = [t for t in model.tyres if pyo.value(model.use_stint[t]) > 0.5]
            num_stints = len(used_tyres)
            num_pitstops = max(0, num_stints - 1)

            print(f"Stints used    : {num_stints}")
            print(f"Pit stops      : {num_pitstops}")
            print("\nStint details:")

            for t in model.tyres:
                if pyo.value(model.use_stint[t]) > 0.5:
                    stint_laps = [
                        l for l in model.L
                        if pyo.value(model.run_lap[t, l]) > 0.5
                    ]

                    stint_length = len(stint_laps)
                    stint_time = sum(
                        pyo.value(model.lap_time[t, l])
                        for l in stint_laps
                    )

                    print(f"  {t}:")
                    print(f"    Laps run : {stint_length}")
                    print(f"    Stint time: {stint_time:.2f} s")

        else:
            print("Cannot find optimal strategy")

    def extract_solution(self, model):
        strategy = []

        for t in model.tyres:
            if pyo.value(model.use_stint[t]) > 0.5:
                stint_laps = [
                    l for l in model.L
                    if pyo.value(model.run_lap[t, l]) > 0.5
                ]

                stint_length = len(stint_laps)
                stint_time = sum(
                    pyo.value(model.lap_time[t, l])
                    for l in stint_laps
                )

                tyre_info = self.get_tyre_info(t)
                curr_idx = len(strategy)

                strategy.append({
                    "tyre_id": t,
                    "stint_number": 1 + curr_idx,
                    "driver_number": 0,
                    "lap_start": 1 if curr_idx == 0 else (strategy[curr_idx-1]["lap_end"] + 1),
                    "lap_end": (0 if curr_idx == 0 else strategy[curr_idx-1]["lap_end"]) + stint_length,
                    "compound": tyre_info.compound,
                    "tyre_age_at_start": tyre_info.init_age,
                })

        return strategy
    
    def get_results(self, results, model):
        if results.solver.termination_condition != "optimal":
            return None, None

        # ---------- Race-level summary ----------
        total_time = pyo.value(model.obj)

        used_tyres = [
            t for t in model.tyres
            if pyo.value(model.use_stint[t]) > 0.5
        ]

        num_stints = len(used_tyres)
        num_pitstops = max(0, num_stints - 1)

        race_summary = {
            "track": self.track,
            "race_laps": self.race_laps,
            "pit_stop_time": self.pit_stop_overhead,
            "n_stints": num_stints,
            "n_pitstops": num_pitstops,
            "total_race_time": total_time,
        }

        # ---------- Stint-level table ----------
        stint_rows = []
        
        for t in model.tyres:
                if pyo.value(model.use_stint[t]) > 0.5:
                    stint_laps = [
                        l for l in model.L
                        if pyo.value(model.run_lap[t, l]) > 0.5
                    ]

                    stint_length = len(stint_laps)
                    stint_time = sum(
                        pyo.value(model.lap_time[t, l])
                        for l in stint_laps
                    )
            
                    stint_rows.append({
                        "track": self.track,
                        "tyre_id": t,
                        "laps": stint_length,
                        "stint_time": stint_time,
                    })
        return race_summary, stint_rows