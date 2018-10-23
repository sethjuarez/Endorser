import numpy as np
import pandas as pd
import azureml.dataprep as dprep

def ndcg(pred: np.ndarray, truth: np.ndarray) -> np.float64:
    """ Normalized Discounted Cumulative Gain

    Arguments:

    pred -- numpy array of predicted ratings

    truth -- numpy array of true ratings

    Summary:
    It is assumed that the ratings are returned in descending
    order (order matters with this formulation). size of pred 
    and truth must be equal
    """
    assert (len(pred) == len(truth)), "Mismatch encountered in prediction and baseline sizes" 
    ideal = np.log2(np.arange(len(truth)) + 2)
    return np.sum(pred / ideal) / np.sum(truth / ideal)

def query_ratings(dataflow: dprep.api.dataflow.Dataflow, userId: int, topn: int) -> np.ndarray:
    """ Get top N ratings for user

    Arguments:

    dataflow -- azureml.dataprep.api.dataflow.Dataflow

    userId -- int (error if does not exist)

    topn -- top n ratings to return
    """
    ff = dataflow.filter(expression=dprep.col('userId') == userId)
    ff = ff.drop_columns(columns=['userId', 'movieId'])
    ff = ff.sort_desc(columns='rating')
    # reshape into single vector
    x = ff.head(topn).values.T
    assert len(x[0]) == topn, "Invalid return length for query"
    return x[0]

if __name__ == "__main__":
    x = np.ones(20) * 5
    pred = np.array([4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5])
    print(ndcg(pred, x))
