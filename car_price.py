import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
import scipy.cluster.hierarchy as sch


#importing data

df = pd.read_csv('CarPrice_Assignment.csv')


le = preprocessing.LabelEncoder()

for col in df.columns:
    le.fit(df[col])
    df[col]=le.transform(df[col])

#shape

df.shape

df_shape=df.shape
print("Dataset shape:",df_shape)

#sata types

df.dtypes
print("dataTypes :",df.dtypes)

#using descibe as std mean etc

a1= df.describe()  
print(a1)


#Visiulisation
#visualising all feature distributions using pairplot
sns.pairplot(df)
plt.title('Pairplot for the Data', fontsize = 20)
plt.show()


#heatmap
plt.rcParams['figure.figsize'] = (15, 8)
sns.heatmap(df.corr(), cmap = 'Wistia', annot = True)
plt.title('Heatmap for the Data', fontsize = 20)
plt.show()

#new

#

#symbol vs price
plt.figure(figsize=(20,8))
plt.subplot(1,2,1)
plt.title('Symboling versus Price')
sns.boxplot(x=df.symboling, y=df.price, palette=("cubehelix"))

plt.subplot(1,2,2)
plt.title('Symboling Histogram')
sns.countplot(df.symboling, palette=("cubehelix"))

plt.show()


#price&horsepower price&car lenght price&curb weight #price&drive wheel curb weight&horsepower enginesize&boreratio
#visulisation of highwaympg&citympg

plt.rcParams['figure.figsize'] = (18, 8)
plt.subplot(1, 2, 1)
sns.set(style = 'whitegrid')
sns.distplot(df['highwaympg'])
plt.title('Distribution of highwaympg', fontsize = 20)
plt.xlabel('')
plt.ylabel('')


plt.subplot(1, 2, 2)
sns.set(style = 'whitegrid')
sns.distplot(df['citympg'], color = 'red')
plt.title('Distribution of citympg', fontsize = 20)
plt.xlabel('')
plt.ylabel('')
plt.show()


#visulisation car price
#Visualising Price vs variuos parameters

def scatter (x,fig):
    plt.subplot(5,2,fig)
    plt.scatter(df[x],df['price'])
    plt.title(x+' vs Price')
    plt.ylabel('Price')
    plt.xlabel(x)


plt.figure(figsize=(15,30))

scatter('carlength', 1)
scatter('carwidth', 2)
scatter('carheight', 3)
scatter('curbweight', 4)

plt.tight_layout()

#
plt.figure(figsize=(20, 15))
plt.subplot(3,3,1)
sns.boxplot(x = 'doornumber', y = 'price', data = df)
plt.subplot(3,3,2)
sns.boxplot(x = 'fueltype', y = 'price', data = df)
plt.subplot(3,3,3)
sns.boxplot(x = 'aspiration', y = 'price', data = df)
plt.subplot(3,3,4)
sns.boxplot(x = 'carbody', y = 'price', data = df)
plt.subplot(3,3,5)
sns.boxplot(x = 'enginelocation', y = 'price', data = df)
plt.subplot(3,3,6)
sns.boxplot(x = 'drivewheel', y = 'price', data = df)
plt.subplot(3,3,7)
sns.boxplot(x = 'enginetype', y = 'price', data = df)
plt.subplot(3,3,8)
sns.boxplot(x = 'cylindernumber', y = 'price', data = df)
plt.subplot(3,3,9)
sns.boxplot(x = 'fuelsystem', y = 'price', data = df)
plt.show()


# average price of each make
plt.figure(figsize=(25, 6))

plt.subplot(1,3,1)
plt1 = df['cylindernumber'].value_counts().plot(kind='bar')
plt.title('Number of cylinders')
plt1.set(xlabel = 'Number of cylinders', ylabel='Frequency of Number of cylinders')

plt.subplot(1,3,2)
plt1 = df['fueltype'].value_counts().plot(kind='bar')
plt.title('Fuel Type')
plt1.set(xlabel = 'Fuel Type', ylabel='Frequency of Fuel type')

plt.subplot(1,3,3)
plt1 = df['carbody'].value_counts().plot(kind='bar')
plt.title('Car body')
plt1.set(xlabel = 'Car Body', ylabel='Frequency of Car Body')



#visulisation of highwaympg,citympg,horsepower,enginesize,curbweight,carwidth

fig,axes = plt.subplots(2,3,figsize=(18,15))
col = ['highwaympg','citympg','horsepower','enginesize','curbweight','carwidth']
for seg,col in enumerate(col):
    x,y = seg//3,seg%3
    an=sns.scatterplot(x=col, y='price' ,data=df, ax=axes[x,y])
    plt.setp(an.get_xticklabels(), rotation=45)
   
plt.subplots_adjust(hspace=0.5)
plt.show()


#allocating attributes as annual income and spending score and age as target 

x = df.iloc[:,0:24].values
y= df.iloc[:,25]


# checking if there is any NULL data
df.isnull().any().any()


#testing and training data

x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.3,random_state=None)



#Creating a Model
#Linear Regression


linreg = LinearRegression()
linreg.fit(x_train,y_train)
l_reg = linreg.score(x_test,y_test)
y_pred = linreg.predict(x_test)


#r2 score


r2_score(y_test, y_pred)
r2lr=r2_score(y_test, y_pred)
print("r2 score is:",r2lr)

#mse


square = mean_squared_error(y_test,y_pred,squared=False)
print("mean squared error:",square)

#cross valid for lin reg

model_cvl = LinearRegression()
cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
result_cvl = cross_val_score(model_cvl,x,y,cv=cv)
print("the value of cross validation is:",result_cvl.mean())
print("avg cross validation is:{:.2f}".format((result_cvl.mean())))

#scatterplot y-test and y-pred of Linear Regression
fig=plt.figure()
fig= plt.figure(figsize=(6,6))
ax=fig.add_subplot(1,1,1)
ax.scatter((y_test),(y_pred))
plt.title('y_test and y_pred after Linear reg')
plt.xticks(rotation=90)
plt.show()


#Model 2
#Using randomforest regressor for mse

RandomForestRegModel = RandomForestRegressor()
RandomForestRegModel.fit(x_train,y_train)
pred = RandomForestRegModel.predict(x_test)
mse = mean_squared_error(y_test,pred)
rmse = np.sqrt(mse)
print(rmse)

#using random forest for r2 score


model_score = RandomForestRegModel.score(x_test,y_test)
print("coefficient of determination R^2 of the prediction:", model_score)

#using cross validation 

model_cvr = RandomForestRegressor()
result_cvr = cross_val_score(model_cvr,x,y,cv=cv)
print("the value of cross validation is:",result_cvr.mean())
print("avg cross validation is:{:.2f}".format((result_cvr.mean())))

#scatterplot y-test and pred of Random Forest regressor 

fig=plt.figure()
fig= plt.figure(figsize=(6,6))
ax=fig.add_subplot(1,1,1)
ax.scatter((y_test),(pred))
plt.title('y_test and pred for random forest)')
plt.xticks(rotation=90)
plt.show()

## ridge reg

ridgeReg = Ridge(alpha=0.05, normalize=True)
ridgeReg.fit(x_train,y_train)
predreg = ridgeReg.predict(x_test)
#calculating mse
msereg = mean_squared_error(y_test,predreg)
rmse1 = np.sqrt(msereg)
print(rmse1)

r2_score(y_test, predreg)
r2reg=r2_score(y_test, predreg)
print("r2 score is:",r2reg)

#cross validation
model_ridge = ridgeReg
result_ridge = cross_val_score(model_ridge,x,y,cv=cv)
print("the value of cross validation is:",result_ridge.mean())
print("avg cross validation is:{:.2f}".format((result_ridge.mean())))
#scatter plot fo ridge y_test and predreg
fig=plt.figure()
fig= plt.figure(figsize=(6,6))
ax=fig.add_subplot(1,1,1)
ax.scatter((y_test),(predreg))
plt.title('y_test and predreg after Ridge')
plt.xticks(rotation=90)
plt.show()

#lasso reg
lassoReg = Lasso(alpha=0.3, normalize=True)

lassoReg.fit(x_train,y_train)

predlas = lassoReg.predict(x_test)
mselas = mean_squared_error(y_test,predlas)
rmse2 = np.sqrt(mselas)
print(rmse2)

r2_score(y_test, predlas)
r2las=r2_score(y_test, predlas)
print("r2 score is:",r2las)

#cross valid
model_lass = lassoReg
result_lass = cross_val_score(model_lass,x,y,cv=cv)
print("the value of cross validation is:",result_lass.mean())
print("avg cross validation is:{:.2f}".format((result_lass.mean())))
#scatter plot for lasso y_test and predlas
fig=plt.figure()
fig= plt.figure(figsize=(6,6))
ax=fig.add_subplot(1,1,1)
ax.scatter((y_test),(predlas))
plt.title('y_test and predlas after las')
plt.xticks(rotation=90)
plt.show()

#elastic reg
ENreg = ElasticNet(alpha=1, l1_ratio=0.5, normalize=False)

ENreg.fit(x_train,y_train)

predelas = ENreg.predict(x_test)
#predelas = lassoReg.predict(x_test)
mseelas= mean_squared_error(y_test,predelas)
rmse3 = np.sqrt(mse)
print(rmse3)

r2_score(y_test, predelas)
r2elas=r2_score(y_test, predelas)
print("r2 score is:",r2elas)
#cross valid for elas
model_elas = ENreg
result_elas = cross_val_score(model_elas,x,y,cv=cv)
print("the value of cross validation is:",result_elas.mean())
print("avg cross validation is:{:.2f}".format((result_elas.mean())))

#scatterplot for elastic y_test and predelas
fig=plt.figure()
fig= plt.figure(figsize=(6,6))
ax=fig.add_subplot(1,1,1)
ax.scatter((y_test),(predelas))
plt.title('y_test and predelas after elas')
plt.xticks(rotation=90)
plt.show()


# Creating model table

models=pd.DataFrame({
        'Models':['Linear','RandomForest','ridgeReg','Lasso','Elastic'],
        'r2score':[r2lr,model_score,r2reg,r2las,r2elas],
        'mean_squared_error':[square,rmse,rmse1,rmse2,rmse3],
        'Cross-validation':[result_cvr.mean(),result_cvl.mean(),result_ridge.mean(),result_lass.mean(),result_elas.mean()]})
print(models)

#Grid Search
#linear reg 

linreg1 =  LinearRegression()
params_dict={'fit_intercept':[True,False], 'normalize':[True,False] ,'copy_X':[True,False],'n_jobs':[int,None]}
model_3=GridSearchCV(estimator =LinearRegression(),param_grid=params_dict,scoring='r2')
model_3.fit(x_train,y_train)
print('score:%4f' % model_3.score(x_test,y_test))
pred2=model_3.predict(x_test)
R2Lr= r2_score(y_test,pred2)
print('R2(LR):%0.2f'% R2Lr)

#Random Forest


no_of_test=[150]
params_dict={'n_estimators':no_of_test,'n_jobs':[-2],'max_depth':[3,5,7,10],'random_state':[0],'max_features':["auto",'sqrt','log2']}
model_3=GridSearchCV(estimator=RandomForestRegressor(),param_grid=params_dict,scoring='r2')
model_3.fit(x_train,y_train)
print('score:%4f' % model_3.score(x_test,y_test))
pred1=model_3.predict(x_test)
R2= r2_score(y_test,pred1)
print('R2 :%0.2f'% R2)

#ridge

params_dict={'alpha':[1.0], 'fit_intercept':[True,False], 'normalize':[False,True], 'copy_X':[True,False], 'max_iter':[None], 
             'tol':[0.001], 'solver':['auto','svd','lsqr'], 
             'random_state':[None]}
model_4=GridSearchCV(estimator = Ridge(),param_grid=params_dict,scoring='r2')
model_4.fit(x_train,y_train)
print('score:%4f' % model_4.score(x_test,y_test))
pred3=model_4.predict(x_test)
R2ridge= r2_score(y_test,pred3)
print('R2(LR):%0.2f'% R2ridge)

#lass

params_dict={'alpha':[1.0], 'fit_intercept':[True,False], 'normalize':[False,True], 
             'copy_X':[True,False], 'max_iter':[1000], 'tol':[0.001],'warm_start':[False,True], 'random_state':[None]}
model_5=GridSearchCV(estimator = Lasso(),param_grid=params_dict,scoring='r2')
model_5.fit(x_train,y_train)
print('score:%4f' % model_5.score(x_test,y_test))
pred4=model_5.predict(x_test)
R2lasso= r2_score(y_test,pred4)
print('R2(LR):%0.2f'% R2lasso)

#elas

params_dict={'alpha':[1.0], 'l1_ratio':[0.5],'fit_intercept':[True,False], 
             'normalize':[False,True], 'copy_X':[True,False], 'max_iter':[1000], 'tol':[0.001],
             'selection':['cyclic'],'warm_start':[False,True], 'random_state':[None]}
model_6=GridSearchCV(estimator = ElasticNet(),param_grid=params_dict,scoring='r2')
model_6.fit(x_train,y_train)
print('score:%4f' % model_6.score(x_test,y_test))
pred5=model_6.predict(x_test)
R2elas= r2_score(y_test,pred5)
print('R2(LR):%0.2f'% R2elas)

#grid histogram

plt.hist([pred1,pred2,pred3,pred4,pred5],bins=4,
         color = ['green','red','yellow','blue','orange'],label=['pred1','pred2','pred3','pred4','pred5','pred6'])
plt.title('After tuning with GridSearchCV')
plt.xlabel('GridSearchCV(green=RandomForest,yellow=ridge,blue=lasso,orange=elastic,red=Linear)')
plt.show()

#boxplot result1 both side to side

data = [result_cvl,result_cvr,result_elas,result_lass,result_ridge]
ax = plt.subplots()
plt.xlabel('Linear Regression''Random Forest ''Ridge ''Lasso ''ElasticNet')
plt.ylabel('')
plt.title('cross valid box plot')
plt.boxplot(data)
plt.show()


##model comparison

model1=pd.DataFrame({
        'Models':['Linear','RandomForest','ridgeReg','Lasso','Elastic'],
        'r2score':[r2lr,model_score,r2reg,r2las,r2elas],
        'mean_squared_error':[square,rmse,rmse1,rmse2,rmse3],
        'Cross-validation':[result_cvr.mean(),result_cvl.mean(),result_ridge.mean(),result_lass.mean(),result_elas.mean()],
        'Grid R2 score':[R2Lr,R2,R2ridge,R2lasso,R2elas]})
print(model1)

