from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

breast_cancer = load_breast_cancer(as_frame=True)
df = breast_cancer.frame
print(df.to_string())
print(df.info())
print(df.isnull().sum())


print('------preprocessing------')
print('-----FEATURE SCALING-----')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X = df.drop('target',axis=1)
y = df['target']
s = StandardScaler()
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
X_train_scaled = s.fit_transform(X_train)
X_test_scaled = s.transform(X_test)
print('------Decision tree classifier------')
from sklearn.tree import DecisionTreeClassifier , plot_tree
dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)
y_pred = dt.predict(X_test)
from sklearn.metrics import accuracy_score,classification_report,silhouette_score
print('Accuracy:\n',accuracy_score(y_test,y_pred))
print('Classificaton report:\n',classification_report(y_test,y_pred))
print('---Pruned Decision Tree---')
dt_pruned = DecisionTreeClassifier(max_depth=5,min_samples_split=10,random_state=42)
dt_pruned.fit(X_train, y_train)
y_pred_pruned = dt_pruned.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred_pruned))
print('classification report:\n',classification_report(y_test, y_pred_pruned))
print('visualisation')
plt.figure(figsize=(10,10))
plot_tree(dt_pruned,feature_names=X.columns,class_names=["No Disease", "Disease"],filled=True)
plt.show()
print('---Linear regression---')
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)
y_pred_lin_reg = lin_reg.predict(X_test_scaled)
print('r2_score:',r2_score(y_test,y_pred_lin_reg))
print('Mean Square Error:',mean_squared_error(y_test,y_pred_lin_reg))
print('Mean Absolute Error:', mean_absolute_error(y_test,y_pred_lin_reg))
print('----RandomForest----')
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train,y_train)
y_pred_rf = rf.predict(X_test)
print('ACCURACY SCORE',accuracy_score(y_test,y_pred_rf))
print('Classification report:')
print(classification_report(y_test,y_pred_rf))
print('---Support Vector Machine---')
from sklearn.svm import SVC
svm_model = SVC(kernel = 'rbf',C=1,gamma = 'scale')
svm_model.fit(X_train_scaled,y_train)
svm_pred = svm_model.predict(X_test_scaled)
print("SVM Accuracy:",accuracy_score(y_test,svm_pred))
print("classification report:\n",classification_report(y_test,svm_pred))
print('------KMeans Clustering------')
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 2, random_state = 42)
y_pred_clusters = kmeans.fit_predict(X_train_scaled)
print('Silhouette score:\n',silhouette_score(X_train_scaled,y_pred_clusters))


''' by analysing the scores support vector classifier worked the best with an accuracy of 98%'''
