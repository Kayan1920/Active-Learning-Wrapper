from time import time
from tqdm import tqdm
import numpy as np
from skactiveml.utils import unlabeled_indices, labeled_indices, MISSING_LABEL
from skactiveml.pool import UncertaintySampling, GreedySamplingX

VALID_QUERY_STRATEGIES = ['UncertaintySampling', 'GreedySamplingX']

class ActiveLearner:
    def __init__(self, X, y_true, model, query_strategy_type, random_state=0):
        self.X = X
        self.y_true = y_true
        self.model = model
        self.query_strategy_type = query_strategy_type
        self.random_state = random_state

        error = self.ensure_setup()
        if error:
            raise Exception(f"Encountered error while setting up experiment : {error}.")
        
        self.y_annotated = np.full(shape=y_true.shape, fill_value=MISSING_LABEL)
        self.model.fit(self.X, self.y_annotated)
        self.setup_query_strategy()
    
    def ensure_setup(self):
        '''
        Args :
            None

        Returns:
            error (str) : Error message if setup is incorrect.
            None : None if no errors
        
        This function ensures that the setup is functioning as it should.
        '''

        n_samples_x = self.X.shape[0]
        
        if self.y_true.shape[0] != n_samples_x:
            return f"Shape of training examples vs labels not matching. {n_samples_x} vs {self.y_true.shape[0]}."
        
        if self.query_strategy_type not in VALID_QUERY_STRATEGIES:
            return f"{self.get_query_indices} is not a supported query strategy type."

        return None
    
    def setup_query_strategy(self):
        if self.query_strategy_type == 'UncertaintySampling':
            self.query_strategy = UncertaintySampling(method='entropy',
                                                      random_state=self.random_state)
        else:
            self.query_strategy = GreedySamplingX(metric="euclidean",
                                                      random_state=self.random_state)
        
    
    def run_experiment(self, ex_params):
        n_cycles = ex_params['n_cycles']
        batch_size = ex_params['batch_size']
        PRINT_LOGS = ex_params['PRINT_LOGS']
        log_every = ex_params['log_every']
        USE_TQDM = ex_params['USE_TQDM']

        ex_results = {
            'acc_scores' : [],
            'percentage_unlabeled' : [],
            'percentage_labeled' : [],
            'number_of_iterations' : n_cycles
        }

        if (n_cycles * batch_size) > self.X.shape[0]:
            error = f"Not enough trainings samples to label {self.batch_size} batches for {n_cycles} training rounds."
            raise Exception(error)
        
        if USE_TQDM:
            cycle_iterator = tqdm(range(n_cycles))
        else:
            cycle_iterator = range(n_cycles)
        
        ex_start_time = time()
        for cycle_no in cycle_iterator:
            selected_indices = self.get_query_indices(batch_size)
            self.y_annotated[selected_indices] = self.y_true[selected_indices]
            unlbld_idx = unlabeled_indices(self.y_annotated)
            lbld_idx = labeled_indices(self.y_annotated)

            self.model.fit(self.X[lbld_idx], self.y_annotated[lbld_idx])

            acc_score = self.model.score(self.X, self.y_true)
            percentage_unlabeled = 100 * len(unlbld_idx) / self.X.shape[0]
            percentage_labeled = 100 - percentage_unlabeled

            ex_results['acc_scores'].append(acc_score)
            ex_results['percentage_labeled'].append(percentage_labeled)
            ex_results['percentage_unlabeled'].append(percentage_unlabeled)

            if PRINT_LOGS and ((cycle_no % log_every) == 0):
                time_taken = time() - ex_start_time
                print(f'Iteration No - {cycle_no + 1} / {n_cycles} :')
                print(f'\tAccuracy score - {acc_score}')
                print(f'\tPercentage of records labeled - {percentage_labeled}')
                print(f'Total time taken - {time_taken}s | {time_taken / n_cycles}s per iteration')
        
        total_time_taken = time() - ex_start_time
        ex_results['experiment_duration'] = total_time_taken
        return ex_results

    def get_query_indices(self, batch_size=1):
        if self.query_strategy_type == 'UncertaintySampling':
            return self.query_strategy.query(X=self.X,
                                            y=self.y_annotated,
                                            clf=self.model,
                                            batch_size=batch_size,
                                            fit_clf=False)
        else:
            return self.query_strategy.query(X=self.X,
                                            y=self.y_annotated,
                                            batch_size=batch_size)



    