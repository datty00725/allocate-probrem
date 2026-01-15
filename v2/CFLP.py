import cplex
import numpy as np
from itertools import chain, combinations

class SCFLPSolver:
    def __init__(
        self,
        demand_points,
        candidate_sites,
        alpha,
        beta,
        p,
        r,
        h_i,
        J_L,
        J_F,
    ):
        """
        Initializes all data for the S-CFLP instance and precomputes
        distances, w_ij, Ui_L, Ui_F, etc.

        Parameters
        ----------
        demand_points : list of tuple
            Demand points (coordinates), e.g. [(2, 22), (42, 6), ...].
        candidate_sites : list of tuple
            Candidate facility sites (coordinates).
        alpha : float
            Alpha parameter for w_ij = exp(alpha - beta * distance).
        beta : float
            Beta parameter for w_ij = exp(alpha - beta * distance).
        p : int
            Number of facilities to open for the Leader.
        r : int
            Number of facilities to open for the Follower.
        h_i : np.array
            Population (or weighting) for each demand point (length = number of demand points).
        J_L : set
            Indices of existing Leader facilities.
        J_F : set
            Indices of existing Follower facilities.
        """
        self.demand_points = demand_points
        self.candidate_sites = candidate_sites
        self.alpha = alpha
        self.beta = beta
        self.p = p
        self.r = r
        self.h_i = h_i
        self.J_L = J_L
        self.J_F = J_F

        # Basic dimensions
        self.J = len(candidate_sites)  # Number of candidate sites
        self.D = len(demand_points)    # Number of demand points

        # Precompute distances and wij
        self.distances = self.compute_distances(demand_points, candidate_sites)
        self.wij_matrix = self.compute_wij_matrix(self.distances, self.alpha, self.beta)

        # Precompute Leader/Follower utilities from existing facilities
        self.Ui_L = self.compute_Ui_L(self.wij_matrix, self.J_L)
        self.Ui_F = self.compute_Ui_F(self.wij_matrix, self.J_F)

        # Variables that were previously global
        self.F = []             # List of CPLEX problems in the branch-and-cut tree
        self.BestSol = None     # Best solution found so far
        self.theta_LB = 0       # Lower bound on objective
        self.existing_cuts = set()  # To store any submodular cuts already added

    ############################################################################
    # Basic subroutines: distances, w_ij, utilities, objective, etc.
    ############################################################################
    def compute_distances(self, demand_points, candidate_sites):
        """
        Compute distances between each demand point and each candidate site.
        Returns a D x J matrix of distances.
        """
        D = len(demand_points)
        J = len(candidate_sites)
        distances = np.zeros((D, J))
        for d in range(D):
            for j in range(J):
                distances[d, j] = np.sqrt(
                    (demand_points[d][0] - candidate_sites[j][0]) ** 2
                    + (demand_points[d][1] - candidate_sites[j][1]) ** 2
                )
        return distances

    def compute_wij_matrix(self, distances, alpha=0, beta=0.1):
        """
        Compute the w_ij matrix using w_ij = exp(alpha - beta * distance).
        Returns a D x J matrix.
        """
        return np.exp(alpha - beta * distances)

    def compute_Ui_L(self, wij_matrix, J_L):
        """
        Compute Ui^L, the influence of existing Leader facilities on each demand point.
        """
        D, _ = wij_matrix.shape
        if not J_L:
            return np.zeros(D)
        # Sum over columns j in J_L
        return wij_matrix[:, list(J_L)].sum(axis=1)

    def compute_Ui_F(self, wij_matrix, J_F):
        """
        Compute Ui^F, the influence of existing Follower facilities on each demand point.
        """
        D, _ = wij_matrix.shape
        if not J_F:
            return np.zeros(D)
        # Sum over columns j in J_F
        return wij_matrix[:, list(J_F)].sum(axis=1)

    def compute_L(self, h_i, Ui_L, Ui_F, wij, x, y):
        """
        Computes L(x, y) = Σ (over i) of  h_i * ( Ui^L + Σ w_ij x_j ) / (Ui^L + Ui^F + Σ w_ij max{x_j, y_j} )
        """
        numerator = Ui_L + (wij @ x)
        denominator = Ui_L + Ui_F + (wij @ np.maximum(x, y))
        return np.sum(h_i * (numerator / denominator))

    ############################################################################
    # Branch-and-Cut main driver
    ############################################################################
    def solve_s_cflp(self):
        """
        Main Branch-and-Cut framework for solving the S-CFLP.
        """
        # 1. Initialization
        self.initialize_problem()

        # 2. Branch-and-Cut loop
        while self.F:
            problem = self.F.pop()

            # (a) Solve continuous relaxation
            incumbent_solution = self.solve_relaxation(problem)

            # Debugging output
            print("\n=== 🔍 Debugging Relaxation Solution ===")
            print(f"  θ_hat: {incumbent_solution['theta']:.4f}")
            print(f"  x_hat: {incumbent_solution['x']}")
            print("========================================")

            # (b) Solve separation problem
            y_star = self.separation_problem(incumbent_solution)

            # Debugging output
            print("\n=== 🔍 Debugging Separation Problem ===")
            print(f"  y_star: {y_star}")
            print("=======================================")

            # (c) Check if we should add cut
            if self.should_add_cut(incumbent_solution, y_star):
                print(f"\nafter should_add_cut, problems in F : {len(self.F)}")
                new_problem = self.add_cuts(incumbent_solution, y_star, problem)

                # Show details of the new problem with cuts
                self.print_problem_details(new_problem)
                self.F.append(new_problem)
                print(f"after F.append(new_problem), problems in F : {len(self.F)}")
            else:
                # (d) If solution is integer and better than LB => done
                if self.is_integer_solution(incumbent_solution, self.theta_LB):
                    print(f"\nafter is_integer_solution")
                    self.update_best_solution(incumbent_solution)
                    break
                else:
                    # (e) Perform branching
                    print(f"\nafter else")
                    problem_0, problem_1 = self.branch_and_bound(incumbent_solution)
                    self.F.append(problem_0)
                    self.F.append(problem_1)

    ############################################################################
    # Problem initialization, printing, and relaxation solving
    ############################################################################
    def initialize_problem(self):
        """
        Initialize the Master Problem for the Branch-and-Cut.
        """
        problem = cplex.Cplex()

        # We'll maximize θ
        problem.objective.set_sense(problem.objective.sense.maximize)

        # Create x_j variables
        x_names = [f"x{j}" for j in range(self.J)]
        problem.variables.add(
            names=x_names,
            types=["C"] * self.J,
            lb=[0] * self.J,
            ub=[1] * self.J,
            obj=[0] * self.J
        )

        # Create objective variable θ
        problem.variables.add(
            names=["theta"],
            types=["C"],
            lb=[0],
            ub=[1],
            obj=[1]  # The objective is to maximize this θ
        )

        # Add facility limit constraint: sum(x_j) = p
        problem.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=x_names, val=[1.0] * self.J)],
            senses=["E"],
            rhs=[self.p],
            names=["facility_limit"]
        )

        # Initialize Branch-and-Cut data
        self.F = [problem]
        self.BestSol = None
        self.theta_LB = 0
        self.existing_cuts.clear()

    def print_problem_details(self, problem):
        """
        Print out details of the given CPLEX problem:
        objective function, constraints, and variable ranges.
        """
        print("\n=== 🔍 Debugging New Problem ===")

        # Objective function
        print("\n📌 目的関数:")
        obj_coeffs = problem.objective.get_linear()
        var_names = problem.variables.get_names()
        obj_func_str = " + ".join(
            f"{obj_coeffs[i]:.4f} * {var_names[i]}" 
            for i in range(len(var_names))
        )
        print(f"   max {obj_func_str}")

        # Constraints
        print("\n📌 制約:")
        constraint_names = problem.linear_constraints.get_names()
        constraint_senses = problem.linear_constraints.get_senses()
        constraint_rhs = problem.linear_constraints.get_rhs()
        constraints_exprs = problem.linear_constraints.get_rows()

        sense_map = {"E": "=", "L": "≤", "G": "≥"}
        for i, name in enumerate(constraint_names):
            lhs_expr = " + ".join(
                f"{constraints_exprs[i].val[j]:.4f} * {var_names[constraints_exprs[i].ind[j]]}"
                for j in range(len(constraints_exprs[i].ind))
            )
            sense = sense_map[constraint_senses[i]]
            print(f"   {name}: {lhs_expr} {sense} {constraint_rhs[i]:.4f}")

        # Variable bounds
        print("\n📌 変数の範囲:")
        var_lb = problem.variables.get_lower_bounds()
        var_ub = problem.variables.get_upper_bounds()
        for i, name in enumerate(var_names):
            print(f"   {name}: {var_lb[i]} ≤ {name} ≤ {var_ub[i]}")

        print("==============================\n")

    def solve_relaxation(self, problem):
        """
        Solve the continuous relaxation of the given problem.
        Return the incumbent solution { 'x': np.array, 'theta': float }.
        """
        problem.solve()

        x_values = np.array(
            problem.solution.get_values([f"x{j}" for j in range(self.J)])
        )
        theta_value = problem.solution.get_values("theta")

        return {"x": x_values, "theta": theta_value}

    ############################################################################
    # Separation problem & cut generation
    ############################################################################
    def separation_problem(self, incumbent_solution):
        """
        Solve the Follower problem (Proposition 5) given x_hat.
        Return the best follower's response y_hat.
        """
        x_hat = incumbent_solution["x"]

        # a_i(x_hat) = Ui^L + Σ w_ij x_j
        a_i = self.Ui_L + (self.wij_matrix @ x_hat)

        # w_i^L(x_hat) and w_i^U(x_hat)
        # (These lines are user-defined; adjust as needed if the logic differs.)
        # They are not fully specified in the snippet, so here's a plausible approach:
        # Possibly we need min and max across all j for the expression (Ui^F + w_ij*(1-x_hat[j])).
        # The snippet suggests:
        w_i_terms = self.Ui_F[:, np.newaxis] + self.wij_matrix * (1 - x_hat)
        w_i_L = np.min(w_i_terms, axis=1)  # min across j
        w_i_U = np.max(w_i_terms, axis=1)  # max across j

        # beta(x_hat)
        beta_hat = np.sum(
            self.h_i[:, np.newaxis]
            * (
                (a_i[:, np.newaxis] * self.wij_matrix * (1 - x_hat)) /
                ((a_i[:, np.newaxis] + w_i_U[:, np.newaxis]) * (a_i[:, np.newaxis] + w_i_L[:, np.newaxis]))
            ),
            axis=0
        )

        # Sort in descending order
        sorted_indices = np.argsort(-beta_hat)

        # Pick the top r
        y_hat = np.zeros(self.J)
        y_hat[sorted_indices[: self.r]] = 1

        return y_hat

    def should_add_cut(self, incumbent_solution, y_star):
        """
        Check whether we should add a cut by comparing theta_hat and L(x_hat, y_star).
        If θ_hat > L(x_hat, y_star), we add a cut.
        """
        x_hat = incumbent_solution["x"]
        theta_hat = incumbent_solution["theta"]

        L_value = self.compute_L(self.h_i, self.Ui_L, self.Ui_F, self.wij_matrix, x_hat, y_star)
        return theta_hat > L_value

    ############################################################################
    # Submodular cut routines
    ############################################################################
    def generate_subsets(self, elements):
        """
        Generate all subsets of `elements`.
        WARNING: This is 2^len(elements) in size; only feasible for small sets.
        """
        return chain.from_iterable(combinations(elements, r) for r in range(len(elements)+1))

    def compute_L_Y(self, S, Y):
        """
        compute_L_Y(S, Y) = Σ_i h_i * [ (Ui^L + Σ_{j∈S} w_ij ) / (Ui^L + Ui^F + Σ_{j∈S ∪ Y} w_ij ) ]
        """
        # S, Y are sets of facility indices
        numerator = self.Ui_L + self.wij_matrix[:, list(S)].sum(axis=1)
        denominator = (
            self.Ui_L +
            self.Ui_F +
            self.wij_matrix[:, list(S.union(Y))].sum(axis=1)
        )
        f_i_Y = numerator / denominator
        return np.sum(self.h_i * f_i_Y)

    def compute_rho_Y(self, S, k, Y):
        """
        rho_Y(S, k) = L_Y(S ∪ {k}) - L_Y(S).
        """
        return self.compute_L_Y(S.union({k}), Y) - self.compute_L_Y(S, Y)

    def add_cuts(self, incumbent_solution, y_star, problem):
        """
        Add submodular cuts (Constraint (8) in the snippet) based on Algorithm 1
        to ensure no violation in the submodularity region.
        """
        x_hat = incumbent_solution["x"]
        # theta_hat = incumbent_solution["theta"]  # you could use it if needed

        Y = {j for j in range(self.J) if y_star[j] == 1}
        J_set = set(range(self.J))

        # We try all subsets S to find the most violated cut
        min_cut_value = float("inf")
        best_S = None

        for S in self.generate_subsets(range(self.J)):
            S_set = set(S)

            rho_sum_1 = sum(self.compute_rho_Y(J_set - {k}, k, Y) for k in S_set)
            rho_sum_2 = sum(self.compute_rho_Y(J_set - {k}, k, Y) * x_hat[k] for k in S_set)
            rho_sum_3 = sum(self.compute_rho_Y(S_set, k, Y) * x_hat[k] for k in (J_set - S_set))

            cut_value = self.compute_L_Y(S_set, Y) - rho_sum_1 + rho_sum_2 + rho_sum_3

            if cut_value <= min_cut_value:
                min_cut_value = cut_value
                best_S = S_set

        if best_S is not None:
            x_coeffs_s = {
                k: self.compute_rho_Y(J_set - {k}, k, Y) for k in best_S
            }
            x_coeffs_js = {
                k: self.compute_rho_Y(best_S, k, Y) for k in (J_set - best_S)
            }

            constant_term = self.compute_L_Y(best_S, Y) - sum(
                self.compute_rho_Y(J_set - {k}, k, Y) for k in best_S
            )
            rhs_value = -constant_term

            # Build a unique key for the cut
            cut_key = (
                rhs_value,
                tuple(sorted(x_coeffs_s.items())),
                tuple(sorted(x_coeffs_js.items()))
            )

            # Avoid duplicates
            if cut_key in self.existing_cuts:
                print("🚫 [SKIP] Duplicate Cut Detected! Not Adding Again.")
                return problem

            self.existing_cuts.add(cut_key)

            # Build the cut expression:
            #   -theta + Σ_{k in best_S} rho_Y(J\{k}, k,Y) * x_k + Σ_{k in J\best_S} rho_Y(S, k, Y) * x_k >= rhs_value
            ind = ["theta"]
            val = [-1.0]

            # Coeffs for k in best_S
            for k, coeff in x_coeffs_s.items():
                ind.append(f"x{k}")
                val.append(coeff)

            # Coeffs for k in J\best_S
            for k, coeff in x_coeffs_js.items():
                ind.append(f"x{k}")
                val.append(coeff)

            submodular_cut_expr = cplex.SparsePair(ind=ind, val=val)

            # For debug printing
            constraint_str = "-θ"
            for k, coeff in x_coeffs_s.items():
                constraint_str += f" + ({coeff:.4f})x_{k}"
            for k, coeff in x_coeffs_js.items():
                constraint_str += f" + ({coeff:.4f})x_{k}"
            constraint_str += f" ≥ {rhs_value:.4f}"

            print("\n=== ➕ Added Unique Submodular Cut ===")
            print(f"   {constraint_str}")
            print("==============================\n")

            problem.linear_constraints.add(
                lin_expr=[submodular_cut_expr],
                senses=["G"],
                rhs=[rhs_value],
                names=[f"cut_submodular_{len(problem.linear_constraints.get_names())}"]
            )

        return problem

    ############################################################################
    # Checking integrality and updating best solutions
    ############################################################################
    def is_integer_solution(self, incumbent_solution, theta_LB, tol=1e-5):
        """
        Check if x_hat is integral (0 or 1) within tolerance,
        and if θ_hat > current lower bound.
        """
        if incumbent_solution["theta"] <= theta_LB:
            return False

        x_values = incumbent_solution["x"]
        # Check if all x_j are near 0 or 1
        if np.all(np.abs(x_values - np.round(x_values)) < tol):
            return True
        return False

    def update_best_solution(self, incumbent_solution):
        """
        Update the best solution (BestSol) and the lower bound (theta_LB).
        """
        self.BestSol = incumbent_solution["x"].copy()
        self.theta_LB = incumbent_solution["theta"]

        print("\n=== Best Solution Updated ===")
        print(f"BestSol: {self.BestSol}")
        print(f"θ_LB: {self.theta_LB:.4f}")
        print("==============================\n")

    ############################################################################
    # Branch-and-bound routine
    ############################################################################
    def branch_and_bound(self, incumbent_solution):
        """
        Perform branching on the most fractional x_j.
        Return two subproblems with x_j = 0 and x_j = 1, respectively.
        """
        x_hat = incumbent_solution["x"]

        fractional_indices = np.where((x_hat > 1e-5) & (x_hat < 1 - 1e-5))[0]
        if len(fractional_indices) == 0:
            print("⚠️ No fractional x found. Branching is not needed.")
            return None, None

        # Find the index j whose x_j is furthest from an integer
        j_star = fractional_indices[np.argmax(np.abs(x_hat[fractional_indices] - 0.5))]

        print("\n=== 🔀 Branching on x_{} ===".format(j_star))
        print("   Fractional value: x_{} = {:.4f}".format(j_star, x_hat[j_star]))
        print("==============================\n")

        # Get the last (most recent) problem (but we already popped it off in solve_s_cflp())
        # Instead, we can do problem = self._clone_current_problem(), or
        # store it from the caller. For simplicity, we'll do:
        # We'll assume `problem` was the last known problem.
        # Because we are returning two new CPLEX copies, we do:
        problem_0 = self.ForkProblemWithFixedValue(j_star, 0.0)
        problem_1 = self.ForkProblemWithFixedValue(j_star, 1.0)

        return problem_0, problem_1

    def ForkProblemWithFixedValue(self, j_idx, fixed_value):
        """
        Returns a copy of the last problem in self.F with x_j_idx = fixed_value added as a constraint.
        """
        # If there's no problem in self.F, we can't branch
        if not self.F:
            raise RuntimeError("No available problem to copy in self.F. Cannot branch.")
        base_problem = self.F[-1]  # just look at the last problem

        branched_problem = base_problem.copy()
        # Add x_j = fixed_value
        branched_problem.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=[f"x{j_idx}"], val=[1.0])],
            senses=["E"],
            rhs=[fixed_value],
            names=[f"branch_x_{j_idx}_{int(fixed_value)}"]
        )
        return branched_problem


########################################################################
# Example of usage:
########################################################################
if __name__ == "__main__":

    # Example data
    demand_points = [(2, 22), (42, 6), (48, 50), (32, 40), (16, 10)]
    candidate_sites = [(10, 13), (47, 16), (30, 44), (28, 47), (6, 1)]
    alpha = 0
    beta = 0.1
    p = 2
    r = 2
    D = len(demand_points)
    h_i = np.full(D, 1 / D)
    J_L = set()  # existing Leader facilities
    J_F = set()  # existing Follower facilities

    # Instantiate and solve
    solver = SCFLPSolver(
        demand_points,
        candidate_sites,
        alpha,
        beta,
        p,
        r,
        h_i,
        J_L,
        J_F
    )
    solver.solve_s_cflp()

    # Access the best solution found
    print("Best solution x:", solver.BestSol)
    print("Best solution objective (theta_LB):", solver.theta_LB)
