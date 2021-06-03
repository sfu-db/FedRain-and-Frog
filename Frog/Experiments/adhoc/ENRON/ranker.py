import numpy as np


class LossRanker:
    
    def __init__(self, trainer):
        self.trainer = trainer
        
    def rank(self):
        train_probas = self.trainer.model.predict_proba(self.trainer.X_train)
        correct_class_proba = [row[self.trainer.y_train[num]] for (num,row) in enumerate(train_probas)]
        return np.array(correct_class_proba)
    
    def name(self):
        return "LossRanker"


# Here we calculate the influence of every training point to its own loss
# Remember bugs have high negative influence on their own loss
class LossInfluenceRanker:
    
    def __init__(self, trainer):
        self.trainer = trainer
        
    
    def rank(self):
        predictions = self.trainer.model.predict_proba(self.trainer.X_train)[:,1]
        uncertainty = np.multiply(predictions, 1-predictions)
        
        # Let us calculate the hessians
        hessian_loss = self.trainer.model.C * self.trainer.X_train_dense.T @ np.diag(uncertainty) @ self.trainer.X_train_dense
        hessian_loss += np.eye(hessian_loss.shape[0])
        
        # Calculate grads
        grads = np.diag(predictions - self.trainer.y_train) @ self.trainer.X_train_dense
        right = np.linalg.solve(hessian_loss, grads.T).T
        
        return - np.multiply(grads, right).sum(1)
    
    def name(self):
        return "LossInfluenceRanker"


# Here we calculate the influence of every training point to the total loss
# Even though removing a bug from the training set would increase its loss (see above)
# We also expect that removing it decreases the loss of all other samples
class TotalLossInfluenceRanker:
    
    def __init__(self, trainer):
        self.trainer = trainer
        
    
    def rank(self):
        predictions = self.trainer.model.predict_proba(self.trainer.X_train)[:,1]
        uncertainty = np.multiply(predictions, 1-predictions)
        
        # Let us calculate the hessians
        hessian_loss = self.trainer.model.C * self.trainer.X_train_dense.T @ np.diag(uncertainty) @ self.trainer.X_train_dense
        hessian_loss += np.eye(hessian_loss.shape[0])
        
        # Calculate grads
        grads = np.diag(predictions - self.trainer.y_train) @ self.trainer.X_train_dense
        total = np.sum(grads, 0)
        left = np.linalg.solve(hessian_loss, total.T)
        
        # Observe the sign that we use
        return left @ grads.T
    
    def name(self):
        return "TotalLossInfluenceRanker"
    
class SpamCountComplaintInfluenceRanker:
    
    def __init__(self, trainer):
        self.trainer = trainer
        
    
    def rank(self):
        predictions_train = self.trainer.model.predict_proba(self.trainer.X_train)[:,1]
        uncertainty_train = np.multiply(predictions_train, 1-predictions_train)
        
        # Let us calculate the hessians
        hessian_loss = self.trainer.model.C * self.trainer.X_train_dense.T @ np.diag(uncertainty_train) @ self.trainer.X_train_dense
        hessian_loss += np.eye(hessian_loss.shape[0])
        
        
        # Calculate the complaint grads
        predictions_test = self.trainer.model.predict_proba.predict_proba(self.trainer.X_test)[:,1]
        uncertainty_test = np.multiply(predictions_test, 1-predictions_test)
        
        grad_complaint = np.sum( np.diag(uncertainty_test) @ self.trainer.X_test_dense, 0)
        left = np.linalg.solve(hessian_loss, grad_complaint.T)
        
        # Calculate the training grads
        grads = np.diag(predictions_train - self.trainer.y_train) @ self.trainer.X_train_dense
        
        influence = - left @ grads.T 
        
        test_prediction_hard = self.trainer.model.predict(self.trainer.X_test)
        predicted_spam        = np.sum(test_prediction_hard == 1)
        actual_spam           = np.sum(self.trainer.y_test == 1)
        
        if predicted_spam > actual_spam :
            return - influence
        elif actual_spam < predicted_spam:
            return influence
        else:
            raise Exception
            
    def name(self):
        return "SpamCountComplaintInfluenceRanker"
            
class TiresiasComplaintRanker:
    
    def __init__(self, trainer, X_ex):
        self.trainer = trainer
        self.X_ex    = X_ex
    
    def rank(self):
        # Training uncertainty
        predictions_train = self.trainer.model.predict_proba(self.trainer.X_train)[:,1]
        uncertainty_train = np.multiply(predictions_train, 1-predictions_train)
        
        # Let us calculate the hessians
        hessian_loss = self.trainer.model.C * self.trainer.X_train_dense.T @ np.diag(uncertainty_train) @ self.trainer.X_train_dense
        hessian_loss += np.eye(hessian_loss.shape[0])
        
        # The presumably not spam documents
        predictions_test = self.trainer.model.predict_proba(self.X_ex)[:,1]
        uncertainty_test = np.multiply(predictions_test, 1-predictions_test)
        
        grad_complaint = np.sum( np.diag(uncertainty_test) @ self.X_ex, 0)
        left = np.linalg.solve(hessian_loss, grad_complaint.T)
        
        # Calculate the training grads
        grads = np.diag(predictions_train - self.trainer.y_train) @ self.trainer.X_train_dense
        
        influence = - left @ grads.T 
        
        return - influence
    
    def name(self):
        return "TiresiasComplaintRanker"

class WordComplaintInfluenceRanker:
    
    def __init__(self, trainer, processor, word):
        self.trainer   = trainer
        self.word      = word
        self.processor = processor
    
    def rank(self):
        
        # Training uncertainty
        predictions_train = self.trainer.model.predict_proba(self.trainer.X_train)[:,1]
        uncertainty_train = np.multiply(predictions_train, 1-predictions_train)
        
        # Let us calculate the hessians
        hessian_loss = self.trainer.model.C * self.trainer.X_train_dense.T @ np.diag(uncertainty_train) @ self.trainer.X_train_dense
        hessian_loss += np.eye(hessian_loss.shape[0])
        
        
        # Calculate the complaint grads
        
        # Find all test documents with the word
        word_id                    = self.processor.vectorizer.vocabulary_[self.word]
        targeted_documents         = self.trainer.X_test[:, word_id] > 0
        targeted_documents         = np.squeeze(targeted_documents.toarray())
        test_docs                  = self.trainer.X_test[targeted_documents, :]
        
        predictions_test = self.trainer.model.predict_proba(test_docs)[:,1]
        uncertainty_test = np.multiply(predictions_test, 1-predictions_test)
        
        grad_complaint = np.sum( np.diag(uncertainty_test) @ test_docs.toarray(), 0)
        left = np.linalg.solve(hessian_loss, grad_complaint.T)
        
        # Calculate the training grads
        grads = np.diag(predictions_train - self.trainer.y_train) @ self.trainer.X_train_dense
        
        influence = - left @ grads.T 
        
        test_prediction_hard  = self.trainer.model.predict(test_docs)
        predicted_spam        = np.sum(test_prediction_hard == 1)
        actual_spam           = np.sum(self.trainer.y_test[targeted_documents] == 1)
        
        if predicted_spam > actual_spam :
            return - influence
        elif actual_spam < predicted_spam:
            return influence
        else:
            raise Exception
            
    def name(self):
        return "WordComplaintInfluenceRanker"
    