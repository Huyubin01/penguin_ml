import streamlit as st
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

penguin_df = pd.read_csv('penguins.csv')
penguin_df.dropna(inplace=True)
output = penguin_df['species']
features = penguin_df[['island','bill_length_mm','bill_depth_mm','flipper_length_mm','body_mass_g','sex']]
features = pd.get_dummies(features)
output,uniques = pd.factorize(output)
x_train,x_test,y_train,y_test = train_test_split(features,output,test_size = .8)
rfc = RandomForestClassifier(random_state = 15)
rfc.fit(x_train.values,y_train)
y_pred = rfc.predict(x_test)
score = accuracy_score(y_pred,y_test)
print('Our accuracy score for this model is {}'.format(score))
rf_pickle = open('random_forest_penguin.pickle','wb')
pickle.dump(rfc,rf_pickle)
rf_pickle.close()
output_pickle = open('output_penguin.pickle','wb')
pickle.dump(uniques,output_pickle)
output_pickle.close()
fig, ax = plt.subplots()
ax = sns.barplot(x=rfc.feature_importances_,y=features.columns)
plt.title('Which features are the most importanr for species prediction?')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
fig.savefig('feature_importance.png')
# fig, ax = plt.subplots()
# ax = sns.displot(x=penguin_df['bill_length_mm'],hue = penguin_df['species'])
# plt.axvline(bill_length)
# plt.title('Bill Length by Species')
# st.pyplot(ax)
# fig, ax = plt.subplots()
# ax = sns.displot(x=penguin_df['bill_depth_mm'],hue = penguin_df['species'])
# plt.axvline(bill_depth)
# plt.title('Bill Depth by Species')
# st.pyplot(ax)
# fig, ax = plt.subplots()
# ax = sns.displot(x=penguin_df['flipper_length_mm'],hue = penguin_df['species'])
# plt.axvline(flipper_length)
# plt.title('Flipper Length by Species')
# st.pyplot(ax)
# print('Here are our output variables')
# print(output.head())
# print('Here are our feature variables')
#print(features.head())
#打印前五行数据
#print(penguin_df.head())