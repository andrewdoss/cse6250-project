import csv
import numpy as np
import pickle
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
        A flag indicating whether a training dataset has been indexed.
    h_fitted : bool
        A flag indicating whether an hierarchical model has been fitted.
    f_fitted : bool
        A flag indicating whether a flat model has been fitted.
    """
    def __init__(self, node_desc, node_relations):
        self.nodes = {'root': Node('root', 'The root node.')}
        self.has_index = False
        self.h_fitted = False
        self.f_fitted = False
        with open(node_desc, 'r') as f: # Build nodes with descriptions
            f = csv.reader(f)
            for line in f:
                self.nodes[line[0]] = Node(line[0], line[1])

        with open(node_relations, 'r') as f: # Build node-node relations
            f = csv.reader(f)
            for line in f:
                self.nodes[line[1]].children.append(self.nodes[line[0]])
                self.nodes[line[0]].parent = self.nodes[line[1]]

        # Recursively compute node depths
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
        """Returns node object, if it exists.

        node : String
            The string representation of the code corresponding to a node.
        """
        try:
            return self.nodes[node]
        except KeyError:
            return None

    def index_df(self, df, codes='fcode'):
        """Build an index from codes to dataframe rows.

        df : pandas DataFrame
            A pandas dataframe containing training data for model fitting.
        codes : String
            The name of the column containing ICD-9 code labels as a ";"
            delimited String.
        """
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

    def fit_hmodel(self, X, parent='root', model=LinearSVC,
                   model_params = dict(), max_depth=None):
        """Fit a hierarchical model

        X : 2d nd-array
            Training feature array aligning with dataframe used for index.
        parent : Node object, optional
            Where to start fitting from, default is root.
        model : an estimator class, optional
            The estimator class to use for the hierarchical model, default
            LinearSVC.
        model_params : dictionary, optional
            Hyperparameters for the selected estimator, default to defaults.
        max_depth : int
            Max depth for fitting the hierarchical model.
        """
        assert self.has_index, "Must first build index for tree."

        # Debugging counter list, can remove later if desired
        count_list = [0]

        def fit_one_v_all_hmodel(X, parent, model, model_params, count_list):
            """Recursively fit one v. all models with relevent samples.

            parent: Node object
                A Node object to start building the hierarchy from.
            """
            # Check if max fitting depth has been reached
            if max_depth is None or max_depth >= parent.depth + 1:
                # Check if node has children for fitting one v. all models
                if len(parent.children) > 0:
                    for child in parent.children:
                        # Recurse down through children
                        fit_one_v_all_hmodel(X, child, model, model_params,
                                             count_list)
                        # Fit one v. all model for each child
                        child.fit_node_hmodel(X, self.get_rows(parent),
                                              self.get_rows(child), model,
                                              model_params)
                        # Count number of models fits attempted
                        count_list[0] += 1
            return None

        if parent == 'root':
            parent = self.root
        fit_one_v_all_hmodel(X, parent, model, model_params, count_list)
        print(f'{count_list[0]} model fits attempted.')
        self.h_fitted = True
        return None

    def predict_hmodel(self, X, start='root', pred_parents=False):
        """Make predictions using hierarchical models.

        X : 2d nd-array
            Testing feature array.
        start : Node object, optional
            Where to start predicting from. It is assumed that all samples in
            X are positive at the starting node, default is the root.

        This implementation is bad, lots of for loops. Can be vectorized
        using more thoughtful data structures and Pandas/NumPy.
        """
        assert self.h_fitted, "Must fit hierarchical models first."

        # Create list of lists to hold all positive labels for each row
        preds = [[] for i in range(X.shape[0])]

        def predict_one_v_all_hmodel(X, parent, pos_at_parent, preds,
                                     pred_parents):
            """Recursively predict for each node with relevent samples.

            X : 2d nd-array
                Test samples.
            parent: Node object
                A Node object to start predicting from.
            pos_at_parent : 1d nd-array, len = X.shape[0]
                Array tracking which samples were positive at parent.
            preds : list of lists
                Used to accumulate positive labels for each sample.
            """
            # Check if node has children for predictions
            if len(parent.children) > 0:
                for child in parent.children:
                    # Check if model exists
                    if child.hmodel is not None:
                        if child.hmodel == 1:
                            child_preds = pos_at_parent
                        else:
                            # This is a prototype, but very inefficient to calc
                            # on all samples even though many are not relevent.
                            # I'm only doing this for now because it makes
                            # managing indices easier.
                            child_preds = (child.hmodel.predict(X)
                                           + pos_at_parent == 2.0).astype(int)
                        # Store positive results if label in training set
                        # or if pred_parents is True
                        if len(child.rows) > 0 or pred_parents:
                            for idx in list(np.nonzero(child_preds)[0]):
                                preds[idx].append(child.code)
                        # If any positive results, recurse through child
                        if np.sum(child_preds) > 0:
                            predict_one_v_all_hmodel(X, child, child_preds,
                                                     preds, pred_parents)

            return None

        if start == 'root':
            start = self.root
        predict_one_v_all_hmodel(X, start, np.ones(X.shape[0]), preds,
                                 pred_parents)

        return preds

    def fit_fmodel(self, X, parent='root', model=LinearSVC, model_params=dict(),
                   max_depth=None, fit_parents=False):
        """Fit a flat model (train each model on all data)

        X : 2d nd-array
            Training feature array aligning with dataframe used for index.
        model : an estimator class, optional
            The estimator class to use for the flat model, default
            LinearSVC.
        model_params : dictionary, optional
            Hyperparameters for the selected estimator, default to defaults.
        max_depth : int
            Max depth for fitting the flat model.
        fit_parents : bool
            Whether to fit models for parents of labels, or only actual labels.
        """
        assert self.has_index, "Must first build index for tree."

        # Debugging counter list, can remove later if desired
        count_list = [0]

        def fit_one_v_all_fmodel(X, parent, model, model_params,
                                 count_list, fit_parents):
            """Recursively fit one v. all models with all samples.

            parent: Node object
                A Node object to start building the hierarchy from.
            """
            # Check if max fitting depth has been reached
            if max_depth is None or max_depth >= parent.depth + 1:
                # Check if node has children for fitting one v. all models
                if len(parent.children) > 0:
                    for child in parent.children:
                        # Recurse down through children
                        fit_one_v_all_fmodel(X, child, model, model_params,
                                             count_list, fit_parents)
                        # Fit one v. all model for each child if the child is a
                        # label in the training set or if pred_parents selected
                        if len(child.rows) > 0 or fit_parents:
                            child.fit_node_fmodel(X, self.get_rows(child), model,
                                                  model_params)
                            # Count number of models fits attempted
                            count_list[0] += 1
            return None

        if parent == 'root':
            parent = self.root
        fit_one_v_all_fmodel(X, parent, model, model_params, count_list,
                             fit_parents)
        print(f'{count_list[0]} model fits attempted.')
        self.f_fitted = True
        return None

    def predict_fmodel(self, X, start='root', pred_parents=False):
        """Make predictions using flat models.

        X : 2d nd-array
            Testing feature array.
        start : Node object, optional
            Where to start predicting from.

        This implementation is bad, lots of for loops. Can be vectorized
        using more thoughtful data structures and Pandas/NumPy.
        """
        assert self.f_fitted, "Must fit flat models first."

        # Create list of lists to hold all positive labels for each row
        preds = [[] for i in range(X.shape[0])]

        def predict_one_v_all_fmodel(X, parent, preds, pred_parents):
            """Recursively predict for each node with relevent samples.

            X : 2d nd-array
                Test samples.
            parent: Node object
                A Node object to start predicting from.
            preds : list of lists
                Used to accumulate positive labels for each sample.
            """
            # Check if node has children
            if len(parent.children) > 0:
                for child in parent.children:
                    # Check if model exists and if prediction should be made
                    if child.fmodel is not None and (len(child.rows) > 0 or
                                                     pred_parents):
                        child_preds = child.fmodel.predict(X)
                        # Store positive results
                        for idx in list(np.nonzero(child_preds)[0]):
                            preds[idx].append(child.code)
                    # Recurse through child
                    predict_one_v_all_fmodel(X, child, preds, pred_parents)

            return None

        if start == 'root':
            start = self.root
        predict_one_v_all_fmodel(X, start, preds, pred_parents)

        return preds

    def save_models(self, file, model_type='h'):
        """Helper for serializing models.

        This is the most straightforward approach, but may be faster and use
        less disk space to just write the model weights to a .csv file.
        """
        models = {}
        for code, node in self.nodes.items():
            if model_type == 'h':
                model = node.hmodel
                if model is not None:
                    models[code] = model
            if model_type == 'f':
                model = node.fmodel
                if model is not None:
                    models[code] = model
        with open(file, 'wb') as dump_file:
            pickle.dump(models, dump_file)

        return None


    def load_models(self, file, model_type='h'):
        """Helper for deserializing models.

        This is the most straightforward approach, but may be faster and use
        less disk space to just write the model weights to a .csv file.
        """

        with open(file, 'rb') as recover_file:
            models = pickle.load(recover_file)

        for code, model in models.items():
            if model_type == 'h':
                self.get_node(code).hmodel = model
            if model_type == 'f':
                self.get_node(code).fmodel = model

        if model_type == 'h':
            self.h_fitted = True
        elif model_type == 'f':
            self.f_fitted = True

        return None

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
        self.hmodel = None
        self.fmodel = None

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

    def fit_node_hmodel(self, X, parent_rows, child_rows, model, model_params):
        """Fit model using rows corresponding to hierarchical approach."""
        temp_X, temp_y = X[parent_rows,:], np.zeros(X.shape[0])
        temp_y[child_rows] = 1.0
        temp_y = temp_y[parent_rows]
        # If all examples are positive, implement constant model
        if temp_y.shape[0] > 0 and temp_y.mean() == 1.0:
            self.hmodel = 1
        # Can only fit models if there are positive examples
        elif temp_y.sum() > 0:
            self.hmodel = model(**model_params)
            self.hmodel.fit(temp_X, temp_y)
        return None

    def fit_node_fmodel(self, X, child_rows, model, model_params):
        """Fit model using rows corresponding to hierarchical approach."""
        temp_X, temp_y = X, np.zeros(X.shape[0])
        temp_y[child_rows] = 1.0
        # Can only fit models if there are positive examples
        if temp_y.sum() > 0:
            self.fmodel = model(**model_params)
            self.fmodel.fit(temp_X, temp_y)
        return None
