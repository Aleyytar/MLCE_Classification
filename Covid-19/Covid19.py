import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler


sns.set()
plt.style.use("seaborn-v0_8-darkgrid")

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import confusion_matrix, f1_score
## dataseti Okuma
df = pd.read_csv("Covid Data.csv")
##dataseti satır ve colums sayısı yazdır
##print("Shape of df :",df.shape)
## dataseti satır ve sutun sayısını terminalde ayarla
pd.set_option('display.max_columns', None) 
pd.set_option('display.max_rows', 10) 
#yazdırma komutu
# print(df.head())
###print(df.info())

#boş data varmı bakıyoruz 
##print(df.isnull().sum())


#Let us get the count of pregenancies grouped by sex to check whether 97, 98 relates to gender.
# print(df.groupby(['PREGNANT','SEX'])['SEX'].count())

##datesetinde 97 98 sayıları pregnant columda bulunmakta 
#fakat olmaması gerekiyor bizde bu yüzde hamile olanlara 1 olmayanlara 2 degerini vericez


df.PREGNANT = df.PREGNANT.replace(97, 2) #Replace 97 with not pregnant.
df.PREGNANT = df.PREGNANT.replace(98, 2)


##Bu kod, bir veri çerçevesindeki (df) "PREGNANT" (gebelik) sütununda 97 ve 98 değerlerini değiştiriyor.

# İlk satırda df.PREGNANT = df.PREGNANT.replace(97, 2) ifadesi, "PREGNANT" sütunundaki değerleri değiştiriyor. replace() fonksiyonu, belirtilen değeri 

# (97) başka bir değerle (2) değiştirir.

# Bu satır, 97 değerini (hamilelik durumunu ifade eden bir kod) 2 ile değiştirerek hamile olmayanları temsil etmek üzere bu veriyi güncelliyor.

# İkinci satırda ise df.PREGNANT = df.PREGNANT.replace(98, 2) ifadesi, aynı işlemi 98 değeri için yapıyor.

# Yani, 98 değerini de 2 ile değiştirerek hamile olmayanları temsil eden bir kod olarak değiştiriyor.

# Sonuç olarak, bu kod, veri çerçevesindeki "PREGNANT" sütunundaki 97 ve 98 değerlerini hamile olmayanları temsil eden 2 değeri ile değiştiriyor.

df = df[(df.PNEUMONIA == 1) | (df.PNEUMONIA == 2)]
df = df[(df.DIABETES == 1) | (df.DIABETES == 2)]
df = df[(df.COPD == 1) | (df.COPD == 2)]
df = df[(df.ASTHMA == 1) | (df.ASTHMA == 2)]
df = df[(df.INMSUPR == 1) | (df.INMSUPR == 2)]
df = df[(df.HIPERTENSION == 1) | (df.HIPERTENSION == 2)]
df = df[(df.OTHER_DISEASE == 1) | (df.OTHER_DISEASE == 2)]
df = df[(df.CARDIOVASCULAR == 1) | (df.CARDIOVASCULAR == 2)]
df = df[(df.OBESITY == 1) | (df.OBESITY == 2)]
df = df[(df.RENAL_CHRONIC == 1) | (df.RENAL_CHRONIC == 2)]
df = df[(df.TOBACCO == 1) | (df.TOBACCO == 2)]

# print(f'shape of dataset: - {df.shape}')


##yogun bakım da ve entübe ünitesinde yatan hast sayısı 
print(df['ICU'].value_counts())
print(df['INTUBED'].value_counts())
##ıcu ve ıntubed columlarını delete ettik
df.drop(['ICU', 'INTUBED'], axis=1, inplace=True)
# print(df.shape)


## ölüm tarihini öldü mü binary sınıflandırma 
df['DEATH'] = [2 if row=='9999-99-99' else 1 for row in df['DATE_DIED']]
# print(df['DEATH'].value_counts())

# print(df.shape)

# print(df['DEATH'])


#date_died kaldıma işlemi ihtiyacımız kalmadı lineer işlemleri için yapabilmek için objeden kurtulmamız lazım 
df.drop('DATE_DIED', axis=1, inplace=True)



X = df.drop('DEATH', axis=1)
y = df['DEATH']

sampler = RandomUnderSampler(random_state=42)
X_sampled, y_sampled = sampler.fit_resample(X, y)
print(y_sampled.value_counts())


X_train, X_test, y_train, y_test = train_test_split(X_sampled, y_sampled, test_size=0.2, random_state=42)
print('X train :', X_train.shape)
print('X test :', X_test.shape)
print('y train :', y_train.shape)
print('y test :', y_test.shape)


def scaling(data):
    scaler = StandardScaler()
    data['AGE'] = scaler.fit_transform(data.loc[:,['AGE']])
    return data

X_train_scaled = scaling(X_train)
X_test_scaled = scaling(X_test)




def metric(algorithm_name, y_true, preds):
    '''
    Function that generates a report on the performance of the classification model.
    param : algorithm_name :- Name of the algorithm.
    param : y_true         :- The actual labels of the test data.
    param : preds          :- The predicted label of the train data.
    '''
    print(f'Classification report of {algorithm_name}')
    print('='*50)
    cm = confusion_matrix(y_true, preds)
    F1 = f1_score(y_true, preds)
    total = sum(sum(cm))
    accuracy = (cm[0,0] + cm[1,1])/total
    sensitivity = cm[0,0]/(cm[0,0] + cm[0,1])
    specificity = cm[1,1]/(cm[1,0]+cm[1,1])
    print(f'\nThe confusion Matrix is :\n{cm}')
    print(f'\nThe F1 Score is         : {F1}')
    print(f'\nThe accuracy is         : {accuracy}')
    print(f'\nThe sensitivity is      : {sensitivity}')
    print(f'\nThe specificity is      : {specificity}')
    print('='*50)


# 3.1 Logistic Regression.
logistic = LogisticRegression()
logistic.fit(X_train_scaled, y_train)
preds_lr = logistic.predict(X_test_scaled)
metric('Logistic Regression', y_test, preds_lr)