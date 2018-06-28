import numpy as np
import cv2
import os
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn import manifold
from sklearn import neighbors
from sklearn import cross_validation
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn import metrics,datasets
from sklearn import model_selection
import seaborn as sns
import tensorflow as tf

sns.set_style(style='white')
def detections(images_path,label,labelints):
    save_path = "datasets\ourdatasets\{label1}save".format(label1=label)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    features=[]
    labels=[]
    labelsint=[]
    faceDet=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    happy_images = os.listdir(images_path)
    print(save_path)
    for image_name in happy_images:
        image_full_path = '{p}\{i}'.format(p=images_path, i=image_name)
        image_save_path= '{p}\{i}'.format(p=save_path, i=image_name)
        img = cv2.imread(image_full_path)
    
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face = faceDet.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in face:
            face_part = gray[y:y+h,x:x+w]
            height, width = face_part.shape
            scale=0.1
            res = cv2.resize(face_part,(10,10), interpolation = cv2.INTER_AREA )
            cv2.imwrite( image_save_path,res)
            features.append(res)
            labels.append(label)
            labelsint.append(labelints)
            '''
            cv2.imshow('res',res)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            '''
    return features, labels,labelsint

def plot_roc_curve(classifier, data, targets, classifier_name="SVM"):
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    target_names = {'happy': 0, 'sadness':1, 'angry': 2}
    target_colors = {'happy': 'red', 'sadness':'blue', 'angry': 'green'}
    i = 0
    for target_name in target_names:
        targets_binary = targets[:]
        for i, target in enumerate(targets):
            if target==target_name:
                targets_binary[i]=0
            else:
                targets_binary[i]=1
            
        x_train, x_test, y_train, y_test = model_selection.train_test_split(data,targets_binary, test_size=0.33, random_state=0)
        y_predicted = []
        if classifier_name=="SVM":
            y_predicted = classifier.fit(x_train, y_train).decision_function(x_test)
        else:
            y_predicted = classifier.fit(x_train, y_train).predict(x_test)
        
        fpr[target_name], tpr[target_name], _ = roc_curve(y_test, y_predicted)
        roc_auc[target_name] = auc(fpr[target_name], tpr[target_name])
        plt.plot(fpr[target_name], tpr[target_name], 'k-', lw=2, color=target_colors[target_name], label=target_name)
        i+=1
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for '+ classifier_name)
    plt.legend(loc="lower right")
    sns.despine(top=True, right=True)
    plt.savefig("roc_curve"+classifier_name+".png")
    #plt.show()
    plt.close()
def plot_scores(scores, classifier_name):
    folds = [i for i in range(1,len(scores)+1)]
    ax = sns.barplot(folds, scores*100)
    ax.set(ylabel="Accuracy (%)")
    sns.despine(top=True, right=True)
    plt.title("5-Fold Cross Validation " + classifier_name)
    #plt.show()
    plt.savefig("crossvalidation_"+classifier_name+".png")
    plt.close()
    
def KNN(data,targets):
    n_neighbors = 4
    kNNClassifier = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
    scores = cross_validation.cross_val_score( kNNClassifier, data, targets, cv=5)
    
    plot_scores(scores, classifier_name="KNN")
    print(targets)
    print("KNN - Cross validation")
    print(scores)
    print(np.mean(scores))
    
    plot_roc_curve(kNNClassifier, data, targets, classifier_name="KNN")
    
def SVM(data,targets):
    clf_svm = LinearSVC()
    scores = cross_validation.cross_val_score(clf_svm, data, targets, cv=5)
    print("SVm - Cross validation")
    print(scores)
    print(np.mean(scores))
    plot_scores(scores, classifier_name="SVM")
    plot_roc_curve(clf_svm, data, targets)
    
''' 
def DNN(data,targets):
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        data,targets, test_size=0.2, random_state=42)
    
    # Build 3 layer DNN with 10, 20, 10 units respectively.
    feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(
        x_train)
    classifier = tf.contrib.learn.DNNClassifier(
        feature_columns=feature_columns, hidden_units=[10, 20, 10], n_classes=3)
    
    # Fit and predict.
    classifier.fit(x_train, y_train, steps=200)
    predictions = list(classifier.predict(x_test, as_iterable=True))
    score = metrics.accuracy_score(y_test, predictions)
    print('Accuracy: {0:f}'.format(score))'''

if __name__ == "__main__":
    features=[]
    labels=[]
    labelsint=[]
    happyface="datasets\ourdatasets\happiness"
    sadnessface="datasets\ourdatasets\sadness"
    angryface=r"datasets\ourdatasets\angry"
    happyFeatures,happyLabel,happylabelint=detections(happyface,'happy',0)
    sadnessFeatures,sadnessLabel,sadlabelint=detections(sadnessface,'sadness',1)
    angryFeatures,angryLabel,angrylabelint=detections(angryface,'angry',2)
    features.extend(happyFeatures)
    features.extend(sadnessFeatures)
    features.extend(angryFeatures)
    labels.extend(happyLabel)
    labels.extend(sadnessLabel)
    labels.extend(angryLabel)
    
    labelsint.extend(happylabelint)
    labelsint.extend(sadlabelint)
    labelsint.extend(angrylabelint)
    features=np.array(features)
    #print(labels)
    #print(features)
    #print(np.array(happyFeatures).shape)
    #print(np.array(sadnessFeatures).shape)
    #print(np.array(angryFeatures).shape)
    #print(features.shape)
    n_samples = len(features)
    print(n_samples)
    images_reshaped=features.reshape((n_samples, -1))
    pca = PCA(n_components=3)
    allData = pca.fit_transform(images_reshaped)
    colors = ("green", "blue", "red")
    #plt.scatter(allData[:, 0], allData[:, 1], allData[:, 2], color=colors, label=labels)
    
    
    #plt.show()
    
    #plt.scatter(allData[:,0], allData[:,1], color=colors, label=labels)
    #plt.show()
    #plt.scatter(allData[:,0], allData[:,1])
    #plt.show('figure1.png')
    KNN(allData, labels)
    SVM(allData,labels)
   # DNN(images_reshaped, labelsint)
