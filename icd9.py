import csv
import numpy as np
from sklearn.svm import LinearSVC

class ICD9Tree(object):
    """A tree representation of the ICD-9 codes.

    Parameters
    ----------
    node_desc : String
        The filepath for a tab-delimited node-description file.
    node_relations : String
        The filepath for a tab-delimited parent-child relations file.

    Attributes
    ----------
    nodes : dict
        A dictionary of key = code, value = Node object.
    has_index : bool
        A flag indicating whether a dataframe has been indexed.
    hsvm_fitted : bool
        A flag indicating whether an hsvm has been fitted.
    """
    def __init__(self, node_desc, node_relations):
        self.nodes = {'root': Node('root', 'The root node.')}
        self.index = False
        self.fitted = False
        with open(node_desc, 'r') as f: # Build nodes map
            f = csv.reader(f)
            for line in f:
                self.nodes[line[0]] = Node(line[0], line[1])

        with open(node_relations, 'r') as f: # Build node relations
            f = csv.reader(f)
            for line in f:
                self.nodes[line[1]].children.append(self.nodes[line[0]])
                self.nodes[line[0]].parent = self.nodes[line[1]]

        # Populate depths recursively
        def assign_depths(node, depth=0):
            node.depth = depth
            depth += 1
            if len(node.children) > 0:
                for child in node.children:
                    assign_depths(child, depth)
            else:
                return None

        assign_depths(self.nodes['root'])

    def get_node(self, node):
        """Returns node object, if it exists."""
        try:
            return self.nodes[node]
        except KeyError:
            return None

    def index_df(self, df, codes='fcode'):
        """Build an index from codes to dataframe rows."""
        # Fills missing with empty string and dedupes with sets
        codes_s = df[codes].copy()
        codes_s.fillna('', inplace=True)
        codes_s = codes_s.str.split(';').apply(set)

        # Corresponding rows are stored at the node level
        for index, fcodes in codes_s.iteritems():
            for code in fcodes:
                self.get_node(code).rows.append(index)

        # Set flag indicating completed index and return self for chaining
        self.has_index = True
        return None

    def fit_hsvm(self, X, parent='root', max_depth=None):
        """Fit a hierarchical SVM

        X : 2d nd-array
            Training feature array aligning with dataframe used for index.
        parent : Node object, optional
            Where to start fitting from, default is root.
        max_depth : int
            Max depth for fitting the hierarchical SVM.
        """
        assert self.has_index, "Must first build index for tree."

        # Debugging counter list, can remove later
        count_list = [0]

        def fit_one_v_all_hsvm(X, parent, count_list):
            """Recursively fit one v. all SVMs with relevent samples.

            parent: Node object
                A Node object to start building the hierarchy from.
            """
            # Check if max fitting depth has been reached
            if max_depth is None or max_depth >= parent.depth + 1:
                # Check if node has children for fitting one v. all models
                if len(parent.children) > 0:
                    for child in parent.children:
                        # Recurse down through children
                        fit_one_v_all_hsvm(X, child, count_list)
                        # Fit one v. all model for each child
                        child.fit_node_hsvm(X, self.get_rows(parent), self.get_rows(child))
                        # Count number of models fits attempted
                        count_list[0] += 1
            return None

        if parent == 'root':
            parent = self.root
        fit_one_v_all_hsvm(X, parent, count_list)
        print(f'{count_list[0]} model fits attempted.')
        self.fitted = True
        return None

    def predict_hsvm(self, X, start='root'):
        """Make hierarchical predictions using svms

        X : 2d nd-array
            Testing feature array.
        start : Node object, optional
            Where to start predicting from. It is assumed that all samples in
            X are positive at the starting node, default is the root.

        This implementation is very bad, lots of for loops. Can be vectorized
        using more thoughtful data structures and Pandas/NumPy.
        """
        assert self.fitted, "Must fit hsvm models first."

        # Create list of lists to hold all positive labels for each row
        preds = [[] for i in range(X.shape[0])]

        def predict_one_v_all_hsvm(X, parent, pos_at_parent, preds):
            """Recursively predict one v. all SVMs with relevent samples.

            X : 2d nd-array
                Test samples.
            parent: Node object
                A Node object to start predicting from.
            pos_at_parent : 1d nd-array, len = X.shape[0]
                Array tracking which samples were positive at parent.
            preds : list of lists
                Used to accumulate positive labels for each sample.
            """
            # Check if node has children for fitting one v. all models
            if len(parent.children) > 0:
                for child in parent.children:
                    # Check if model exists
                    if child.hsvm is not None:
                        if child.hsvm == 1:
                            child_preds = pos_at_parent
                        else:
                            # This is a prototype, but very inefficient to calc
                            # on all samples even though many are not relevent
                            # I'm only doing this for now because it makes
                            # managing indices easier
                            child_preds = (child.hsvm.predict(X)
                                           + pos_at_parent == 2.0).astype(int)
                        # Store positive results
                        for idx in list(np.nonzero(child_preds)[0]):
                            preds[idx].append(child.code)
                        # If any positive results, recurse through child
                        if np.sum(child_preds) > 0:
                            predict_one_v_all_hsvm(X, child, child_preds, preds)

            return None

        if start == 'root':
            start = self.root
        predict_one_v_all_hsvm(X, start, np.ones(X.shape[0]), preds)

        return preds

    def serialize_hsvm(self, file):
        """Helper for serializing hsvm models."""
        pass

    def deserialize_hsvm(self, file):
        """Helper for deserializing hasvm models."""
        pass

    def get_rows(self, node):
        """Returns rows corresonding to node and its descendents"""
        desc = node.descendents
        rows = set()
        for d in desc:
            rows = rows.union(d.rows)
        return sorted(list(rows))

    @property
    def root(self):
        """Returns the root node"""
        return self.nodes['root']

class Node(object):
    """A node in the ICD-9 tree."""
    def __init__(self, code, descr):
        self.code = code
        self.descr = descr
        self.parent = None
        self.children = []
        self.depth = None
        self.rows = []
        self.hsvm = None

    def __repr__(self):
        return "icd9.Node(code={})".format(self.code)

    def __str__(self):
        return self.code

    @property
    def descendents(self):
        """Returns all descendent nodes, including self."""

        # Populate descendents recursively
        def get_desc(node, descendents):
            descendents.append(node)
            if len(node.children) > 0:
                for child in node.children:
                    get_desc(child, descendents)
            else:
                return None

        descendents = []
        get_desc(self, descendents)

        return descendents

    @property
    def leaves(self):
        """Returns all descendent leaves, including self."""
        # Populate leaves recursively
        def get_leaves(node, leaves):
            if len(node.children) > 0:
                for child in node.children:
                    get_leaves(child, leaves)
            else:
                leaves.append(node)
                return None

        leaves = []
        get_leaves(self, leaves)

        return leaves

    def fit_node_hsvm(self, X, parent_rows, child_rows):
        """Fit svm using rows corresponding to hierarchical approach."""
        temp_X, temp_y = X[parent_rows,:], np.zeros(X.shape[0])
        temp_y[child_rows] = 1.0
        temp_y = temp_y[parent_rows]
        # If all examples are positive, implement constant model
        if temp_y.shape[0] > 0 and temp_y.mean() == 1.0:
            self.hsvm = 1
        # Can only fit models if there are positive examples
        elif temp_y.sum() > 0:
            self.hsvm = LinearSVC(C=1.0)
            self.hsvm.fit(temp_X, temp_y)
        return None
