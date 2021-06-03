from ranker import WordComplaintInfluenceRanker
from ranker import TotalLossInfluenceRanker
from ranker import LossInfluenceRanker
from ranker import TiresiasComplaintRanker
from ranker import LossRanker
import numpy as np
from time import time
from tqdm import tnrange
from json import dumps

from trainer import Trainer
from fixer import AutoFixer
from fixer import OracleFixer

from sklearn.linear_model import LogisticRegression


class ComplaintManager:

    def __init__(self, processor, word):
        # Find all test documents with the word
        word_id = processor.vectorizer.vocabulary_[word]
        targeted_documents = processor.X_test[:, word_id] > 0
        targeted_documents = np.squeeze(targeted_documents.toarray())
        self.test_docs = processor.X_test[targeted_documents, :]
        self.actual_spam = np.sum(processor.y_test[targeted_documents] == 1)

    def query_result(self, trainer):
        test_prediction_hard = trainer.model.predict(self.test_docs)
        predicted_spam = np.sum(test_prediction_hard == 1)
        return predicted_spam

    def complaint(self):
        return self.actual_spam


def process(ranker, fixer, elapses, complaint, query_results):
    result = {
        "proc": "ENRON",
        "seed": 1,
        "params": dumps({"ranker": ranker.name(),
                         "fixer": fixer.name()}),
        "elapses": np.asarray(elapses).astype(float).tolist(),
        "deletions": np.asarray(fixer.deletions).astype(int).tolist(),
        "truth": fixer.truth.astype(bool).tolist(),
        "complaint": float(complaint),
        "query_results": np.asarray(query_results).astype(float).tolist()
    }

    return result


def auto_debug(processor, word, Ranker):

    X_train,  ycrptd = processor.corrupt_by_word(word)
    trainer = Trainer(X_train,  ycrptd, processor.X_test, processor.y_test)
    ranker = Ranker(trainer)
    fixer = AutoFixer(ycrptd != processor.y_train)
    manager = ComplaintManager(processor, word)

    correction_rounds = 2 * np.sum(ycrptd != processor.y_train)

    trainer.train()

    elapses = []
    query_results = []

    now = time()

    for i in tnrange(correction_rounds):
        try:
            rank = ranker.rank()
        except Exception:
            break

        correct_guess = fixer.fix(trainer, rank)

        elapses.append(time() - now)
        query_results.append(manager.query_result(trainer))

        trainer.train()

    return process(ranker, fixer, elapses, manager.complaint(), query_results)


def oracle_debug(processor, word, Ranker):

    X_train,  ycrptd = processor.corrupt_by_word(word)
    trainer = Trainer(X_train,  ycrptd, processor.X_test, processor.y_test)
    ranker = Ranker(trainer)
    fixer = OracleFixer(ycrptd != processor.y_train)
    manager = ComplaintManager(processor, word)

    correction_rounds = 2 * np.sum(ycrptd != processor.y_train)

    trainer.train()
    rerank = True

    elapses = []
    query_results = []

    now = time()

    for i in tnrange(correction_rounds):

        if rerank:
            try:
                rank = ranker.rank()
            except Exception:
                break

        correct_guess = fixer.fix(trainer, rank)

        if correct_guess:
            trainer.train()

        elapses.append(time() - now)
        query_results.append(manager.query_result(trainer))

        rerank = correct_guess > 0

    return process(ranker, fixer, elapses, manager.complaint(), query_results)


def auto_loss(processor, word):
    return auto_debug(processor, word, LossRanker)


def oracle_loss(processor, word):
    return oracle_debug(processor, word, LossRanker)


def auto_loss_influence(processor, word):
    return auto_debug(processor, word, LossInfluenceRanker)


def oracle_loss_influence(processor, word):
    return oracle_debug(processor, word, LossInfluenceRanker)


def auto_total_loss_influence(processor, word):
    return auto_debug(processor, word, TotalLossInfluenceRanker)


def oracle_total_loss_influence(processor, word):
    return oracle_debug(processor, word, TotalLossInfluenceRanker)


def auto_complaint_influence(processor, word):
    def ranker(trainer): return WordComplaintInfluenceRanker(trainer, processor, word)
    return auto_debug(processor, word, ranker)


def oracle_complaint_influence(processor, word):
    def ranker(trainer): return WordComplaintInfluenceRanker(trainer, processor, word)
    return oracle_debug(processor, word, ranker)


def prepare_repair(processor, word):
    model = LogisticRegression(C=1, solver="lbfgs",
                               max_iter=800,
                               fit_intercept=False,
                               warm_start=True)

    X_train,  ycrptd = processor.corrupt_by_word(word)
    model.fit(X_train, ycrptd)

    word_id = processor.vectorizer.vocabulary_[word]
    targeted_documents = processor.X_test[:, word_id] > 0
    targeted_documents = np.squeeze(targeted_documents.toarray())
    test_docs = processor.X_test[targeted_documents, :]

    count_predicted = np.sum(model.predict(test_docs) == 1)
    count_actual = np.sum(processor.y_test[targeted_documents] == 1)

    if count_predicted > count_actual:
        print(f'Larger count {count_predicted} > {count_actual}')
        difference = count_predicted - count_actual
        condition = np.logical_and(targeted_documents, model.predict(processor.X_test) == 1)
        pool_of_mistakes = processor.X_test[condition, :]
        np.random.seed(0)
        examples = np.random.choice(count_predicted, difference, replace=False)
        X_ex = pool_of_mistakes[examples, :].toarray()
    else:
        raise Exception

    return X_ex


def auto_baseline(processor, word):
    repair = prepare_repair(processor, word)
    def ranker(trainer): return TiresiasComplaintRanker(trainer, repair)
    return auto_debug(processor, word, ranker)


def oracle_baseline(processor, word):
    repair = prepare_repair(processor, word)
    def ranker(trainer): return TiresiasComplaintRanker(trainer, repair)
    return oracle_debug(processor, word, ranker)
