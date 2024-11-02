import streamlit as st
import joblib
st.title('IRIS 品種預測')
svm = joblib.load('svm_model.joblib')
knn = joblib.load('KNN_model.joblib')
lr = joblib.load('LR_model.joblib')
rf = joblib.load('RF_model.joblib')


s1 = st.sidebar.selectbox("### 請選擇模型:", ("SVM", "KNN", "LogisticRegression", "RandomForest"))
if s1 == "SVM":
    model = svm
elif s1 == "KNN":
    model = knn
elif s1 == "LogisticRegression":
    model = lr
elif s1 == "RandomForest":
    model = rf
st.write(s1)


se1 = st.slider('花萼長度:',3.0,8.0,5.8)
se2 = st.slider('花萼寬度:',1.8,5.0,3.8)
se3 = st.slider('花瓣長度:',0.7,7.2,5.0)
se4 = st.slider('花瓣寬度:',0.0,3.5,1.8)

labels = ['Setosa', 'Versicolor', 'Virginica']
if st.button('進行預測'):
    x = [[se1,se2,se3,se4]]
    y = model.predict(x)
    st.write(y[0])
    st.write('### 預測結果,品種名稱應是:',labels[y[0]])
