import numpy as np
import itertools
import copy
import time
import queue

class Variable:
    def __init__(self, no_of_runs, start_of_runs, len_of_runs, domain_of_starts):
        self.no_of_runs = no_of_runs
        self.start_of_runs = start_of_runs
        self.len_of_runs = len_of_runs
        self.domain_of_starts = domain_of_starts
        self.must_be_filled = get_must_be_filled(self.domain_of_starts, self.len_of_runs)

    def __str__(self):
        return ("No of Runs=" + str(self.no_of_runs) +
                " ,Len of Runs=" + str(self.len_of_runs) +
                " ,Start of Runs=" + str(self.start_of_runs) +
                "\n Domain Of Starts:" + str(self.domain_of_starts) +
                "\n Must Be Filled:" + str(self.must_be_filled))

def get_must_be_filled(domain_of_starts, lengths):
    must_be_filled_for_every_domain_values = []
    for permutation in domain_of_starts:
        must_be_filled = []
        for i in range(len(permutation)):
            perm_i = permutation[i]
            # print("perm_i:" + str(perm_i))
            len_j = lengths[i]
            for j in range(len_j):
                # print("j:" + str(j))
                # print("adding:" + str(perm_i+j))
                must_be_filled.append(perm_i + j)
                # print("Must Fill:" + str(must_be_filled) + " for this index:" + str(index))
        must_be_filled_for_every_domain_values.append(must_be_filled)
    return must_be_filled_for_every_domain_values


def solve(constraints):
    """
    Implement me!!!!!!!
    This function takes in a set of constraints. The first dimension is the axis
    to which the constraints refer to. The second dimension is the list of constraints
    for some axis index pair. The third demsion is a single constraint of the form 
    [i,j] which means a run of i js. For example, [4,1] would correspond to a block
    [1,1,1,1].
    
    The return value of this function should be a numpy array that satisfies all
    of the constraints.


	A puzzle will have the constraints of the following format:
	
    
	array([
		[list([[4, 1]]), 
		 list([[1, 1], [1, 1], [1, 1]]),
         list([[3, 1], [1, 1]]), 
		 list([[2, 1]]), 
		 list([[1, 1], [1, 1]])],
        [list([[2, 1]]), 
		 list([[1, 1], [1, 1]]), 
		 list([[3, 1], [1, 1]]),
         list([[1, 1], [1, 1]]), 
		 list([[5, 1]])]
		], dtype=object)
	
	And a corresponding solution may be:

	array([[0, 1, 1, 1, 1],
		   [1, 0, 1, 0, 1],
		   [1, 1, 1, 0, 1],
		   [0, 0, 0, 1, 1],
		   [0, 0, 1, 0, 1]])



	Consider a more complicated set of constraints for a colored nonogram.

	array([
	   [list([[1, 1], [1, 4], [1, 2], [1, 1], [1, 2], [1, 1]]),
        list([[1, 3], [1, 4], [1, 3]]), 
		list([[1, 2]]),
        list([[1, 4], [1, 1]]), 
		list([[2, 2], [2, 1], [1, 3]]),
        list([[1, 2], [1, 3], [1, 2]]), 
		list([[2, 1]])],
       [list([[1, 3], [1, 4], [1, 2]]),
        list([[1, 1], [1, 4], [1, 2], [1, 2], [1, 1]]),
        list([[1, 4], [1, 1], [1, 2], [1, 1]]), 
		list([[1, 2], [1, 1]]),
        list([[1, 1], [2, 3]]), 
		list([[1, 2], [1, 3]]),
        list([[1, 1], [1, 1], [1, 2]])]], 
		dtype=object)

	And a corresponding solution may be:

	array([
		   [0, 1, 4, 2, 1, 2, 1],
		   [3, 4, 0, 0, 0, 3, 0],
		   [0, 2, 0, 0, 0, 0, 0],
		   [4, 0, 0, 0, 0, 0, 1],
		   [2, 2, 1, 1, 3, 0, 0],
		   [0, 0, 2, 0, 3, 0, 2],
		   [0, 1, 1, 0, 0, 0, 0]
		 ])


    """
    rows = dim0 = len(constraints[0])
    columns = dim1 = len(constraints[1])
    print("Dim0: " + str(dim0))
    print("Dim1: " + str(dim1))
    np_constraints = np.array(constraints)
    #print(np_constraints)

    row_constraints = constraints[0]
    column_constraints = constraints[1]
    print("Row Constraints: " + str(row_constraints))
    print("Column Constraints: " + str(column_constraints))

    Variables = {}
    Row_Variables = {}
    Col_Variables = {}
    Variables["ROW"] = Row_Variables
    Variables["COL"] = Col_Variables

    #TODO: Reduce to 1 loop that does row as well as columns
    iterator = 0
    for each_run_set in row_constraints:
        print("Each Row Run Set:" + str(each_run_set))
        no_of_runs = len(each_run_set)
        start_of_runs = []
        len_of_runs = []
        domain_of_starts = []
        total_length = 0
        run_lengths = [a[0] for a in each_run_set]
        # print(run_lengths)
        run_iter = 0
        for each_run in each_run_set:
            print("Each Run:" + str(each_run))
            run_length = each_run[0]
            len_of_runs.append(run_length)
            each_domain_of_start = []
            for i in range(total_length, columns - sum(run_lengths[run_iter:]) + 1):
                each_domain_of_start.append(i)
            domain_of_starts.append(each_domain_of_start)
            total_length += run_length
            run_iter += 1
        domains = create_permutations(len_of_runs, domain_of_starts)
        Row_Variables[iterator] = Variable(no_of_runs,start_of_runs,len_of_runs, domains)
        iterator += 1

    iterator = 0
    for each_run_set in column_constraints:
        print("Each Col Run Set:" + str(each_run_set))
        no_of_runs = len(each_run_set)
        start_of_runs = []
        len_of_runs = []
        domain_of_starts = []
        total_length = 0
        for each_run in each_run_set:
            print("Each Run:" + str(each_run))
            run_length = each_run[0]
            len_of_runs.append(run_length)
            each_domain_of_start = []
            for i in range(total_length, rows - run_length + 1):
                each_domain_of_start.append(i)
            domain_of_starts.append(each_domain_of_start)
            total_length += run_length
        domains = create_permutations(len_of_runs, domain_of_starts)
        Col_Variables[iterator] = Variable(no_of_runs, start_of_runs, len_of_runs, domains)
        iterator += 1

    print("\nROWS")
    print_map(Variables["ROW"])
    print("\nCOLUMNS")
    print_map(Variables["COL"])

    start_time = time.time()
    # print("Constraint Prop for an answer...")
    # type, starting_index = select_next_MRV_DH(Variables)
    # print("Will start with: " + type + str(starting_index))
    # reduce_consistencies_with_constraint_propogation(Variables, type, starting_index)
    # print("\nROWS")
    # print_map(Variables["ROW"])
    # print("\nCOLUMNS")
    # print_map(Variables["COL"])
    print("Backtracking for an answer...")
    Final_Variables = Backtrack_WithHeuristics(Variables, rows)
    end_time = time.time()
    print("Found Answer in " + str(end_time-start_time) + " secs")
    print("\nROWS")
    print_map(Final_Variables["ROW"])
    print("\nCOLUMNS")
    print_map(Final_Variables["COL"])

    result = np.zeros((rows, columns), np.int64)
    print(result)
    create_solution(result, Final_Variables, rows)
    print(result)

    return result
        #np.random.randint(2, size=(dim0, dim1))


def create_solution(result, Variables, rows):
    Row_Constraints = Variables["ROW"]
    for i in range(rows):
        curr_row = Row_Constraints[i]
        lengths = curr_row.len_of_runs
        starts = curr_row.domain_of_starts[0]
        for j in range(len(starts)):
            curr_start = starts[j]
            for k in range(lengths[j]):
                current_value = curr_start + k
                result[i][current_value] = 1


def get_minimum_remaining_values_variable(Variables):
    Row_Constraints = Variables["ROW"]
    minimum_size = 9999999999
    minimum_index = -1
    for each_row_key, each_row_value in Row_Constraints.items():
        print("row:" + str(each_row_key) + " values:" + str(remaining_domain_size(each_row_value)))
        if is_value_set(each_row_value):
            continue
        size_of_domain = remaining_domain_size(each_row_value)
        if size_of_domain < minimum_size:
            minimum_size = size_of_domain
            minimum_index = each_row_key
    return minimum_index

def select_next_MRV_DH(Variables):
    Row_Constraints = Variables["ROW"]
    Column_Constraints = Variables["COL"]
    minimum_size = 9999999999
    minimum_index = -1

    for each_row_key, each_row_value in Row_Constraints.items():
        if is_value_set(each_row_value):
            continue
        size_of_domain = remaining_domain_size(each_row_value)
        if size_of_domain < minimum_size:
            minimum_size = size_of_domain
            minimum_index = each_row_key
        elif size_of_domain == minimum_size:
            if get_degree(each_row_value) > get_degree(Row_Constraints[minimum_index]):
                minimum_index = each_row_key

    flag = False
    for each_col_key, each_col_value in Column_Constraints.items():
        if is_value_set(each_col_value):
            continue
        size_of_domain = remaining_domain_size(each_col_value)
        if size_of_domain < minimum_size:
            flag = True
            minimum_size = size_of_domain
            minimum_index = each_col_key
        elif size_of_domain == minimum_size and not flag:
            if get_degree(each_col_value) > get_degree(Row_Constraints[minimum_index]):
                flag = True
                minimum_index = each_col_key
        elif size_of_domain == minimum_size:
            if get_degree(each_col_value) > get_degree(Column_Constraints[minimum_index]):
                flag = True
                minimum_index = each_col_key
    if flag:
        return "COL", minimum_index
    return "ROW", minimum_index

def select_next_MRV(Variables):
    Row_Constraints = Variables["ROW"]
    Column_Constraints = Variables["COL"]
    minimum_size = 9999999999
    minimum_index = -1

    for each_row_key, each_row_value in Row_Constraints.items():
        if is_value_set(each_row_value):
            continue
        size_of_domain = remaining_domain_size(each_row_value)
        if size_of_domain < minimum_size:
            minimum_size = size_of_domain
            minimum_index = each_row_key

    flag = False
    for each_col_key, each_col_value in Column_Constraints.items():
        if is_value_set(each_col_value):
            continue
        size_of_domain = remaining_domain_size(each_col_value)
        if size_of_domain < minimum_size:
            flag = True
            minimum_size = size_of_domain
            minimum_index = each_col_key
    if flag:
        return "COL", minimum_index
    return "ROW", minimum_index


def get_degree(variable):
    return sum(variable.len_of_runs)



def is_value_set(variable):
    return remaining_domain_size(variable) == 1

def remaining_domain_size(variable):
    return len(variable.domain_of_starts)


#TODO: Remove the node constraint from here and add it separately
def create_permutations(lengths, domain_of_starts):
    combinations = []
    all_combinations = list(itertools.product(*domain_of_starts))
    #print("ALL combinations:", end='')
    #print(all_combinations)
    for each_combination in all_combinations:
        if is_node_consistent(lengths, each_combination):
            #print(each_combination)
            combinations.append(each_combination)
    return combinations


def is_node_consistent(lengths, each_combination):
    for i in range(len(each_combination)-1):
        if each_combination[i] + lengths[i] >= each_combination[i+1]:
            return False
    return True

rows_checked = set()
cols_checked = set()

def all_values_are_assigned(Variables):
    for each_row_variable in Variables["ROW"].values():
        if not is_value_set(each_row_variable):
            return False
    for each_column_variable in Variables["COL"].values():
        if not is_value_set(each_column_variable):
            return False
    return True


def Backtrack_WithHeuristicsAndConstraintPropogation(Variables, rows):
    # print("\n\nROWS:", end='')
    # for looper in rows_checked:
    #     print(str(looper) + " ", end='')
    # print("\nCOLS:", end='')
    # for looper in cols_checked:
    #     print(str(looper) + " ", end='')
    if all_values_are_assigned(Variables):
        return Variables
    variable_type, index = select_next_MRV_DH(Variables)
    # print(variable_type + ":" + str(index), end=' ')
    current_variable = Variables[variable_type][index]
    # print(current_variable)
    lengths = current_variable.len_of_runs
    domains = current_variable.domain_of_starts
    flag = False

    permutations_by_constraining_factor = []
    for each_permutation in domains:
        constraining_factor, is_valid = get_constraint_reduction(Variables, variable_type, index, lengths, each_permutation)
        # print(str(each_permutation) + " C factor:" + str(constraining_factor) + " Valid:" + str(is_valid))
        if is_valid:
            permutations_by_constraining_factor.append((constraining_factor, each_permutation))
    # print(permutations_by_constraining_factor)
    permutations_by_constraining_factor.sort()
    permutations_by_constraining_factor.reverse()
    # print(permutations_by_constraining_factor)
    perms = []
    for each_permutation in permutations_by_constraining_factor:
        perms.append(each_permutation[1])
    # print("Perms:" + str(perms))
    # print("Domains:" + str(domains))
    for each_permutation in perms:
        New_Variables_Map = copy.deepcopy(Variables)
        # print("Checking " + str(each_permutation))
        new_domain = [each_permutation]
        new_variable = Variable(current_variable.no_of_runs, [], current_variable.len_of_runs, new_domain)
        New_Variables_Map[variable_type][index] = new_variable
        is_consistent = reduce_consistencies_with_constraint_propogation(New_Variables_Map, variable_type, index)
        #columns_consistent = reduce_column_consistencies(New_Variables_Map, index, lengths, each_permutation)
        if is_consistent:
            if variable_type == "ROW":
                rows_checked.add(index)
            else:
                cols_checked.add(index)
            result = Backtrack_WithHeuristicsAndConstraintPropogation(New_Variables_Map, rows)
            if result:
                return result
    return None


def Backtrack_WithHeuristics(Variables, rows):
    # print("\n\nROWS:", end='')
    # for looper in rows_checked:
    #     print(str(looper) + " ", end='')
    # print("\nCOLS:", end='')
    # for looper in cols_checked:
    #     print(str(looper) + " ", end='')
    if all_values_are_assigned(Variables):
        return Variables
    # if len(rows_checked) == len(Variables["COL"]) and len(cols_checked) == len(Variables["ROW"]):
    #     return Variables
    variable_type, index = select_next_MRV_DH(Variables)
    # print(variable_type + ":" + str(index), end=' ')
    current_variable = Variables[variable_type][index]
    # print(current_variable)
    lengths = current_variable.len_of_runs
    domains = current_variable.domain_of_starts
    flag = False

    permutations_by_constraining_factor = []
    for each_permutation in domains:
        constraining_factor, is_valid = get_constraint_reduction(Variables, variable_type, index, lengths, each_permutation)
        # print(str(each_permutation) + " C factor:" + str(constraining_factor) + " Valid:" + str(is_valid))
        if is_valid:
            permutations_by_constraining_factor.append((constraining_factor, each_permutation))
    # print(permutations_by_constraining_factor)
    permutations_by_constraining_factor.sort()
    permutations_by_constraining_factor.reverse()
    # print(permutations_by_constraining_factor)
    perms = []
    for each_permutation in permutations_by_constraining_factor:
        perms.append(each_permutation[1])
    # print("Perms:" + str(perms))
    # print("Domains:" + str(domains))
    for each_permutation in perms:
        New_Variables_Map = copy.deepcopy(Variables)
        # print("Checking " + str(each_permutation))
        new_domain = [each_permutation]
        new_variable = Variable(current_variable.no_of_runs, [], current_variable.len_of_runs, new_domain)
        New_Variables_Map[variable_type][index] = new_variable
        is_consistent = reduce_consistencies(New_Variables_Map, variable_type, index, lengths, each_permutation)
        #columns_consistent = reduce_column_consistencies(New_Variables_Map, index, lengths, each_permutation)
        if is_consistent:
            if variable_type == "ROW":
                rows_checked.add(index)
            else:
                cols_checked.add(index)
            result = Backtrack_WithHeuristics(New_Variables_Map, rows)
            if result:
                return result
    return None


def Backtrack_WithHeuristicsAndConstraintPropogation(Variables, rows):
    # print("\n\nROWS:", end='')
    # for looper in rows_checked:
    #     print(str(looper) + " ", end='')
    # print("\nCOLS:", end='')
    # for looper in cols_checked:
    #     print(str(looper) + " ", end='')
    if all_values_are_assigned(Variables):
        return Variables
    variable_type, index = select_next_MRV_DH(Variables)
    # print(variable_type + ":" + str(index), end=' ')
    current_variable = Variables[variable_type][index]
    # print(current_variable)
    lengths = current_variable.len_of_runs
    domains = current_variable.domain_of_starts
    flag = False

    permutations_by_constraining_factor = []
    for each_permutation in domains:
        constraining_factor, is_valid = get_constraint_reduction(Variables, variable_type, index, lengths, each_permutation)
        # print(str(each_permutation) + " C factor:" + str(constraining_factor) + " Valid:" + str(is_valid))
        if is_valid:
            permutations_by_constraining_factor.append((constraining_factor, each_permutation))
    # print(permutations_by_constraining_factor)
    permutations_by_constraining_factor.sort()
    permutations_by_constraining_factor.reverse()
    # print(permutations_by_constraining_factor)
    perms = []
    for each_permutation in permutations_by_constraining_factor:
        perms.append(each_permutation[1])
    # print("Perms:" + str(perms))
    # print("Domains:" + str(domains))
    for each_permutation in perms:
        New_Variables_Map = copy.deepcopy(Variables)
        # print("Checking " + str(each_permutation))
        new_domain = [each_permutation]
        new_variable = Variable(current_variable.no_of_runs, [], current_variable.len_of_runs, new_domain)
        New_Variables_Map[variable_type][index] = new_variable
        is_consistent = reduce_consistencies_with_constraint_propogation(New_Variables_Map, variable_type, index)
        #columns_consistent = reduce_column_consistencies(New_Variables_Map, index, lengths, each_permutation)
        if is_consistent:
            if variable_type == "ROW":
                rows_checked.add(index)
            else:
                cols_checked.add(index)
            result = Backtrack_WithHeuristicsAndConstraintPropogation(New_Variables_Map, rows)
            if result:
                return result
    return None


def BacktrackNew(Variables, rows):
    # print("\n")
    # for looper in rows_checked:
    #     print(str(looper) + " ", end='')
    # print("\n-----------MAP---------------")
    # New_Variables_Map = copy.deepcopy(Variables)
    # print_map(New_Variables_Map)
    # print("\n-----------END MAP-----------\n\n")
    if all_values_are_assigned(Variables):
        return Variables
    index = get_minimum_remaining_values_variable(Variables)
    print("Checking index " + str(index))
    current_row_variable = Variables["ROW"][index]
    # print(current_row_variable)
    lengths = current_row_variable.len_of_runs
    domains = current_row_variable.domain_of_starts
    flag = False
    for each_permutation in domains:
        New_Variables_Map = copy.deepcopy(Variables)
        # print("Checking " + str(each_permutation))
        columns_consistent = reduce_column_consistencies(New_Variables_Map, index, lengths, each_permutation)
        if columns_consistent:
            new_domain = [each_permutation]
            new_row_variable = Variable(current_row_variable.no_of_runs, [], current_row_variable.len_of_runs, new_domain)
            New_Variables_Map["ROW"][index] = new_row_variable
            rows_checked.add(index)
            result = BacktrackNew(New_Variables_Map, rows)
            if result:
                return result
    return None


def Backtrack(Variables, index, rows):
    # print("\n")
    # for looper in range(index):
    #     print(str(looper) + " ", end='')
    # print("\n-----------MAP---------------")
    # New_Variables_Map = copy.deepcopy(Variables)
    # print_map(New_Variables_Map)
    # print("\n-----------END MAP-----------\n\n")
    if index == rows:
        return Variables
    current_row_variable = Variables["ROW"][index]
    # print(current_row_variable)
    lengths = current_row_variable.len_of_runs
    domains = current_row_variable.domain_of_starts
    flag = False
    for each_permutation in domains:
        New_Variables_Map = copy.deepcopy(Variables)
        # print("Checking " + str(each_permutation))
        columns_consistent = reduce_column_consistencies(New_Variables_Map, index, lengths, each_permutation)
        if columns_consistent:
            new_domain = [each_permutation]
            new_row_variable = Variable(current_row_variable.no_of_runs, [], current_row_variable.len_of_runs, new_domain)
            New_Variables_Map["ROW"][index] = new_row_variable
            result = Backtrack(New_Variables_Map, index + 1, rows)
            if result:
                return result
    return None




    # if flag:
    #     return Backtrack(New_Variables_Map, index+1, rows)
    # else:
    #     return None
    # for each_domain_start in current_row_variable.domain_of_starts:
    #     print(each_domain_start)


def get_constraint_reduction(Variables, variable_type, index, lengths, permutation):
    size_of_domains = 0
    must_be_filled = []
    for i in range(len(permutation)):
        perm_i = permutation[i]
        # print("perm_i:" + str(perm_i))
        len_j = lengths[i]
        for j in range(len_j):
            # print("j:" + str(j))
            # print("adding:" + str(perm_i+j))
            must_be_filled.append(perm_i + j)
    # print("Must Fill:" + str(must_be_filled) + " for this index:" + str(index))

    # Q = queue.Queue()
    if variable_type == "ROW":
        other_type = "COL"
    else:
        other_type = "ROW"
    for each_value in must_be_filled:
        # print("Checking " + other_type + " " + str(each_value))
        variable = Variables[other_type][each_value]
        # print("Var:" + str(variable))
        domain_of_starts = variable.domain_of_starts
        run_lengths = variable.len_of_runs
        new_domains = []
        for each_combination in domain_of_starts:
            # print("Checking assignment:" + str(each_combination))
            if has_value_at(each_combination, run_lengths, index):
                # print("This has a valid assignment at index:" + str(index))
                new_domains.append(each_combination)
        if not new_domains:
            # print("This " + str(permutation) + " has a has no valid assignments for index: " + str(index))
            return 0, False
        size_of_domains += len(new_domains)
        # if len(new_domains) != len(domain_of_starts):
        #     Q.put(each_combination)
        #variable.domain_of_starts = new_domains
        # print("New Column Variable:" + str(column_variable))
    return size_of_domains, True


def reduce_consistencies_with_constraint_propogation(Variables, variable_type, index):
    Q = queue.Queue()
    Q.put((variable_type, index, Variables[variable_type][index]))

    while not Q.empty():
        current_variable_type, prev_index, current_variable = Q.get()
        if current_variable_type == "ROW":
            other_type = "COL"
        else:
            other_type = "ROW"
        set_of_all = set(range(1, len(Variables[other_type])))
        can_be_filled = set([item for sublist in current_variable.must_be_filled for item in sublist])
        can_not_be_filled = set_of_all.difference(can_be_filled)
        for each_value in can_not_be_filled:
            to_delete = []
            neighbour_variable = Variables[other_type][each_value]

            neighbour_domains = neighbour_variable.domain_of_starts
            neighbour_must_be_filled = neighbour_variable.must_be_filled
            domain_size_before_reduction = len(neighbour_domains)
            for i in range(domain_size_before_reduction):
                curr_must_be_filled = neighbour_must_be_filled[i]
                if prev_index in curr_must_be_filled:
                    to_delete.append(i)

            neighbour_domains_new = [x for i, x in enumerate(neighbour_domains) if i not in to_delete]

            if len(neighbour_domains_new) == 0:
                return False
            if len(neighbour_domains_new) < domain_size_before_reduction:
                Variables[other_type][each_value] = Variable(neighbour_variable.no_of_runs,
                                                             neighbour_variable.start_of_runs,
                                                             neighbour_variable.len_of_runs,
                                                             neighbour_domains_new)
                Q.put((other_type, each_value, Variables[other_type][each_value]))

                # print("PUT:" + str(other_type) + " " + str(neighbour_variable))
    return True


def reduce_consistencies(Variables, variable_type, index, lengths, permutation):
    must_be_filled = []
    for i in range(len(permutation)):
        perm_i = permutation[i]
        # print("perm_i:" + str(perm_i))
        len_j = lengths[i]
        for j in range(len_j):
            # print("j:" + str(j))
            # print("adding:" + str(perm_i+j))
            must_be_filled.append(perm_i + j)
    # print("Must Fill:" + str(must_be_filled) + " for this index:" + str(index))

    # Q = queue.Queue()
    if variable_type == "ROW":
        other_type = "COL"
    else:
        other_type = "ROW"
    for each_value in must_be_filled:
        # print("Checking " + other_type + " " + str(each_value))
        variable = Variables[other_type][each_value]
        # print("Var:" + str(variable))
        domain_of_starts = variable.domain_of_starts
        run_lengths = variable.len_of_runs
        new_domains = []
        for each_combination in domain_of_starts:
            # print("Checking assignment:" + str(each_combination))
            if has_value_at(each_combination, run_lengths, index):
                # print("This has a valid assignment at index:" + str(index))
                new_domains.append(each_combination)
        if not new_domains:
            # print("This " + str(permutation) + " has a has no valid assignments for index: " + str(index))
            return False
        # if len(new_domains) != len(domain_of_starts):
        #     Q.put(each_combination)
        variable.domain_of_starts = new_domains
        # print("New Column Variable:" + str(column_variable))
    return True

def reduce_column_consistencies(Variables, index, lengths, permutation):
    print("lengths:" + str(lengths))
    must_be_filled = []
    for i in range(len(permutation)):
        perm_i = permutation[i]
        # print("perm_i:" + str(perm_i))
        len_j = lengths[i]
        for j in range(len_j):
            # print("j:" + str(j))
            # print("adding:" + str(perm_i+j))
            must_be_filled.append(perm_i+j)
    print("Must Fill columns:" + str(must_be_filled) + " for this index:" + str(index))

    for each_column in must_be_filled:
        print("Checking COLUMN:" + str(each_column))
        column_variable = Variables["COL"][each_column]
        print("Col Var:" + str(column_variable))
        domain_of_starts = column_variable.domain_of_starts
        column_lengths = column_variable.len_of_runs
        new_domains = []
        for each_combination in domain_of_starts:
            print("Checking assignment:" + str(each_combination))
            if has_value_at(each_combination, column_lengths, index):
                print("This has a valid assignment at index:" + str(index))
                new_domains.append(each_combination)
        if not new_domains:
            print("This " + str(permutation) + " has a has no valid assignments for index: " + str(index))
            return False
        column_variable.domain_of_starts = new_domains
        print("New Column Variable:" + str(column_variable))
    return True


def has_value_at(column_starts, column_run_lengths, row_index):
    # print("Row Index:" + str(row_index))
    for i in range(len(column_starts)):
        curr_start = column_starts[i]
        # print("Current start = " + str(curr_start))
        for j in range(column_run_lengths[i]):
            curr_j = j
            # print("Current j = " + str(curr_j))
            current_value = curr_start + curr_j
            # print("Current Value = " + str(current_value))
            if current_value == row_index:
                return True
    return False


def print_map(Dictionary):
    for k, v in Dictionary.items():
        print("\nK:" + str(k) + " ==>  " + str(v))
