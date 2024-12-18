import numpy as np
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import warnings
warnings.filterwarnings('ignore')

from catboost import CatBoostRegressor # type: ignore
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier # type: ignore
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans

# パスでファイルを取得
file_path = "C:/Users/81906/Documents/Python_AI/train_home_credit_default_desk.csv"
file_path2 = "C:/Users/81906/Documents/Python_AI/test_home_credit_default_desk.csv"
file_path3 = "C:/Users/81906/Documents/Python_AI/sample_submission_home_credit.csv"

# CSVを読み込む
train = pd.read_csv(file_path)
test = pd.read_csv(file_path2)
sample_sub = pd.read_csv(file_path3)


# NAME_CONTRACT_TYPEの数値化（Label Encoding）
train["NAME_CONTRACT_TYPE"].replace({'Cash loans': 0, 'Revolving loans': 1}, inplace=True)
test["NAME_CONTRACT_TYPE"].replace({'Cash loans': 0, 'Revolving loans': 1}, inplace=True)

# CODE_GENDERの数値化（Label Encoding）
train["CODE_GENDER"].replace({'M': 0, 'XNA':0,  'F': 1}, inplace=True)
test["CODE_GENDER"].replace({'M': 0, 'XNA':0,  'F': 1}, inplace=True)

# OCCUPATION_TYPEをOne Hot Encoding
train_occupation_type_ohe = pd.get_dummies(train["OCCUPATION_TYPE"]).add_prefix("OCCUPATION_TYPE_")
# 結合して元のOCCUPATION_TYPEは消去
train = pd.concat([train, train_occupation_type_ohe], axis=1)
train.drop('OCCUPATION_TYPE', axis=1, inplace=True)
# testも同様
test_occupation_type_ohe = pd.get_dummies(test["OCCUPATION_TYPE"]).add_prefix("OCCUPATION_TYPE_")
test = pd.concat([test, test_occupation_type_ohe], axis=1)
test.drop('OCCUPATION_TYPE', axis=1, inplace=True)

# ORGANIZATION_TYPEの数値化（Count Encoding）(tutorialを一旦引用)
organization_ce = train["ORGANIZATION_TYPE"].value_counts()
train["ORGANIZATION_TYPE"] = train["ORGANIZATION_TYPE"].map(organization_ce)
test["ORGANIZATION_TYPE"] = test["ORGANIZATION_TYPE"].map(organization_ce)

# NAME_INCOME_TYPEがStudentの行をWorkingに書き換える
train.loc[train["NAME_INCOME_TYPE"] == "Student", "NAME_INCOME_TYPE"] = "Working"
# NAME_INCOME_TYPEがBusinessmanの行をWorkingに書き換える
train.loc[train["NAME_INCOME_TYPE"] == "Businessman", "NAME_INCOME_TYPE"] = "Working"
# NAME_INCOME_TYPEがUnenployedの行をWorkingに書き換える
train.loc[train["NAME_INCOME_TYPE"] == "Unemployed", "NAME_INCOME_TYPE"] = "Working"
# NAME_INCOME_TYPEをone hot encoding
train_name_income_type_ohe = pd.get_dummies(train["NAME_INCOME_TYPE"]).add_prefix("NAME_INCOME_TYPE_")
# 元のデータを消して、one hot encodingを結合
train = pd.concat([train, train_name_income_type_ohe], axis=1)
train.drop('NAME_INCOME_TYPE', axis=1, inplace=True)

# test
# NAME_INCOME_TYPEがStudentの行をWorkingに書き換える
test.loc[test["NAME_INCOME_TYPE"] == "Student", "NAME_INCOME_TYPE"] = "Working"
# NAME_INCOME_TYPEがBusinessmanの行をWorkingに書き換える
test.loc[test["NAME_INCOME_TYPE"] == "Businessman", "NAME_INCOME_TYPE"] = "Working"
# NAME_INCOME_TYPEがUnenployedの行をWorkingに書き換える
test.loc[test["NAME_INCOME_TYPE"] == "Unemployed", "NAME_INCOME_TYPE"] = "Working"
# NAME_INCOME_TYPEがMaternity leaveの行をWorkingに書き換える
test.loc[test["NAME_INCOME_TYPE"] == "Maternity leave", "NAME_INCOME_TYPE"] = "Working"
# NAME_INCOME_TYPEをone hot encoding
test_name_income_type_ohe = pd.get_dummies(test["NAME_INCOME_TYPE"]).add_prefix("NAME_INCOME_TYPE_")
# 元のデータを消して、one hot encodingを結合
test = pd.concat([test, test_name_income_type_ohe], axis=1)
test.drop('NAME_INCOME_TYPE', axis=1, inplace=True)

# NAME_EDUCATION_TYPEはone hot encodingして結合し元のやつを消す
train_name_education_type_ohe = pd.get_dummies(train["NAME_EDUCATION_TYPE"]).add_prefix("NAME_EDUCATION_TYPE_")
train = pd.concat([train, train_name_education_type_ohe], axis=1)
train.drop('NAME_EDUCATION_TYPE', axis=1, inplace=True)

# testも同様
test_name_education_type_ohe = pd.get_dummies(test["NAME_EDUCATION_TYPE"]).add_prefix("NAME_EDUCATION_TYPE_")
test = pd.concat([test, test_name_education_type_ohe], axis=1)
test.drop('NAME_EDUCATION_TYPE', axis=1, inplace=True)

# NAME_FAMILY_STATUSはone hot encodingして結合し元のやつを消す
train_name_family_status_ohe = pd.get_dummies(train["NAME_FAMILY_STATUS"]).add_prefix("NAME_FAMILY_STATUS_")
train = pd.concat([train, train_name_family_status_ohe], axis=1)
train.drop('NAME_FAMILY_STATUS', axis=1, inplace=True)

# testも同様
test_name_family_status_ohe = pd.get_dummies(test["NAME_FAMILY_STATUS"]).add_prefix("NAME_FAMILY_STATUS_")
test = pd.concat([test, test_name_family_status_ohe], axis=1)
test.drop('NAME_FAMILY_STATUS', axis=1, inplace=True)

# NAME_HOUSING_TYPEはone hot encodingして結合し元のやつを消す
train_name_housing_type_ohe = pd.get_dummies(train["NAME_HOUSING_TYPE"]).add_prefix("NAME_HOUSING_TYPE_")
train = pd.concat([train, train_name_housing_type_ohe], axis=1)
train.drop('NAME_HOUSING_TYPE', axis=1, inplace=True)

# testも同様
test_name_housing_type_ohe = pd.get_dummies(test["NAME_HOUSING_TYPE"]).add_prefix("NAME_HOUSING_TYPE_")
test = pd.concat([test, test_name_housing_type_ohe], axis=1)
test.drop('NAME_HOUSING_TYPE', axis=1, inplace=True)

# NAME_INCOME_TYPE_Maternity leave列を消す
train.drop('NAME_INCOME_TYPE_Maternity leave', axis=1, inplace=True)

# NAME_FAMILY_STATUS_Unknown列を消す
train.drop('NAME_FAMILY_STATUS_Unknown', axis=1, inplace=True)

# FLAG_OWN_CARはNの方が多いので欠損値はNで埋める
train["FLAG_OWN_CAR"].fillna("N", inplace=True)
test["FLAG_OWN_CAR"].fillna("N", inplace=True)

# FLAG_OWN_CARを数値化
train["FLAG_OWN_CAR"].replace({'N': 0, 'Y': 1}, inplace=True)
test["FLAG_OWN_CAR"].replace({'N': 0, 'Y': 1}, inplace=True)

# FLAG_OWN_REALTYはYの方が多いので欠損値はYで埋める
train["FLAG_OWN_REALTY"].fillna("Y", inplace=True)
test["FLAG_OWN_REALTY"].fillna("Y", inplace=True)

# FLAG_OWN_REALTYを数値化
train["FLAG_OWN_REALTY"].replace({'N': 0, 'Y': 1}, inplace=True)
test["FLAG_OWN_REALTY"].replace({'N': 0, 'Y': 1}, inplace=True)

# NAME_TYPE_SUITEはUnaccompaniedが最も多いので欠損値はUnaccompaniedで埋める
train["NAME_TYPE_SUITE"].fillna("Unaccompanied", inplace=True)
test["NAME_TYPE_SUITE"].fillna("Unaccompanied", inplace=True)

# NAME_TYPE_SUITEをonehotencodingする(確定？)
train_name_type_suite_ohe = pd.get_dummies(train["NAME_TYPE_SUITE"]).add_prefix("NAME_TYPE_SUITE_")
test_name_type_suite_ohe = pd.get_dummies(test["NAME_TYPE_SUITE"]).add_prefix("NAME_TYPE_SUITE_")
# onehotencodingしたものを元のデータと結合
train = pd.concat([train, train_name_type_suite_ohe], axis=1)
test = pd.concat([test, test_name_type_suite_ohe], axis=1)
# NAME_TYPE_SUITEを消去
train.drop('NAME_TYPE_SUITE', axis=1, inplace=True)
test.drop('NAME_TYPE_SUITE', axis=1, inplace=True)

# AMT_ANNUITYは中央値で補完
train["AMT_ANNUITY"].fillna(train["AMT_ANNUITY"].median(), inplace=True)
test["AMT_ANNUITY"].fillna(test["AMT_ANNUITY"].median(), inplace=True)

train["AMT_GOODS_PRICE"].fillna(train["AMT_GOODS_PRICE"].mean(), inplace=True)
test["AMT_GOODS_PRICE"].fillna(test["AMT_GOODS_PRICE"].mean(), inplace=True)

train['CNT_FAM_MEMBERS'] = train['CNT_FAM_MEMBERS'].fillna(train['CNT_FAM_MEMBERS'].median())
test['CNT_FAM_MEMBERS'] = test['CNT_FAM_MEMBERS'].fillna(test['CNT_FAM_MEMBERS'].median())

train["DAYS_LAST_PHONE_CHANGE"].fillna(train["DAYS_LAST_PHONE_CHANGE"].median(), inplace=True)
test["DAYS_LAST_PHONE_CHANGE"].fillna(test["DAYS_LAST_PHONE_CHANGE"].median(), inplace=True)

# OBS_30_CNT_SOCIAL_CIRCLEの欠損値は全て0で埋める
train["OBS_30_CNT_SOCIAL_CIRCLE"].fillna(0, inplace=True)
test["OBS_30_CNT_SOCIAL_CIRCLE"].fillna(0, inplace=True)

# DEF_30_CNT_SOCIAL_CIRCLEの欠損値は全て0で埋める
train["DEF_30_CNT_SOCIAL_CIRCLE"].fillna(0, inplace=True)
test["DEF_30_CNT_SOCIAL_CIRCLE"].fillna(0, inplace=True)

# OBS_60_CNT_SOCIAL_CIRCLEの欠損値は全て0で埋める
train["OBS_60_CNT_SOCIAL_CIRCLE"].fillna(0, inplace=True)
test["OBS_60_CNT_SOCIAL_CIRCLE"].fillna(0, inplace=True)

# DEF_60_CNT_SOCIAL_CIRCLEの欠損値は全て0で埋める
train["DEF_60_CNT_SOCIAL_CIRCLE"].fillna(0, inplace=True)
test["DEF_60_CNT_SOCIAL_CIRCLE"].fillna(0, inplace=True)

# AMT_REQ_CREDIT_BUREAU_HOURの欠損値は全て0で埋める
train["AMT_REQ_CREDIT_BUREAU_HOUR"].fillna(0, inplace=True)
test["AMT_REQ_CREDIT_BUREAU_HOUR"].fillna(0, inplace=True)

# AMT_REQ_CREDIT_BUREAU_MONの欠損値は全て0で埋める
train["AMT_REQ_CREDIT_BUREAU_MON"].fillna(0, inplace=True)
test["AMT_REQ_CREDIT_BUREAU_MON"].fillna(0, inplace=True)

# AMT_REQ_CREDIT_BUREAU_QRTの欠損値は全て0で埋める
train["AMT_REQ_CREDIT_BUREAU_QRT"].fillna(0, inplace=True)
test["AMT_REQ_CREDIT_BUREAU_QRT"].fillna(0, inplace=True)

# AMT_REQ_CREDIT_BUREAU_YEARの欠損値は全て0で埋める
train["AMT_REQ_CREDIT_BUREAU_YEAR"].fillna(0, inplace=True)
test["AMT_REQ_CREDIT_BUREAU_YEAR"].fillna(0, inplace=True)

# OWN_CAR_AGEの欠損値は全て中央値で埋める
train["OWN_CAR_AGE"].fillna(train["OWN_CAR_AGE"].median(), inplace=True)
test["OWN_CAR_AGE"].fillna(test["OWN_CAR_AGE"].median(), inplace=True)

# OWN_CAR_AGEが60以上の行は中央値に変換
train.loc[train["OWN_CAR_AGE"] >= 60, "OWN_CAR_AGE"] = train["OWN_CAR_AGE"].median()

# NAME_EDUCATION_TYPE_Secondary / secondary specialの名前をNAME_EDUCATION_TYPE_Secondary/secondary_specialに変更
train.rename(columns={'NAME_EDUCATION_TYPE_Secondary / secondary special': 'NAME_EDUCATION_TYPE_Secondary/secondary_special'}, inplace=True)
test.rename(columns={'NAME_EDUCATION_TYPE_Secondary / secondary special': 'NAME_EDUCATION_TYPE_Secondary/secondary_special'}, inplace=True)

# NAME_FAMILY_STATUS_Single / not marriedをNAME_FAMILY_STATUS_Single/not_marriedに変更
train.rename(columns={'NAME_FAMILY_STATUS_Single / not married': 'NAME_FAMILY_STATUS_Single/not_married'}, inplace=True)
test.rename(columns={'NAME_FAMILY_STATUS_Single / not married': 'NAME_FAMILY_STATUS_Single/not_married'}, inplace=True)

# NAME_HOUSING_TYPE_House / apartmentをNAME_HOUSING_TYPE_House/apartmentに変更
train.rename(columns={'NAME_HOUSING_TYPE_House / apartment': 'NAME_HOUSING_TYPE_House/apartment'}, inplace=True)
test.rename(columns={'NAME_HOUSING_TYPE_House / apartment': 'NAME_HOUSING_TYPE_House/apartment'}, inplace=True)

# OCCUPATION_TYPE_Cleaning staffをOCCUPATION_TYPE_Cleaning_staffに変更
train.rename(columns={'OCCUPATION_TYPE_Cleaning staff': 'OCCUPATION_TYPE_Cleaning_staff'}, inplace=True)
test.rename(columns={'OCCUPATION_TYPE_Cleaning staff': 'OCCUPATION_TYPE_Cleaning_staff'}, inplace=True)
# OCCUPATION_TYPE_Cooking staffをOCCUPATION_TYPE_Cooking_staffに変更
train.rename(columns={'OCCUPATION_TYPE_Cooking staff': 'OCCUPATION_TYPE_Cooking_staff'}, inplace=True)
test.rename(columns={'OCCUPATION_TYPE_Cooking staff': 'OCCUPATION_TYPE_Cooking_staff'}, inplace=True)
# OCCUPATION_TYPE_Core staffをOCCUPATION_TYPE_Core_staffに変更
train.rename(columns={'OCCUPATION_TYPE_Core staff': 'OCCUPATION_TYPE_Core_staff'}, inplace=True)
test.rename(columns={'OCCUPATION_TYPE_Core staff': 'OCCUPATION_TYPE_Core_staff'}, inplace=True)
# OCCUPATION_TYPE_HR staffをOCCUPATION_TYPE_HR_staffに変更
train.rename(columns={'OCCUPATION_TYPE_HR staff': 'OCCUPATION_TYPE_HR_staff'}, inplace=True)
test.rename(columns={'OCCUPATION_TYPE_HR staff': 'OCCUPATION_TYPE_HR_staff'}, inplace=True)
# OCCUPATION_TYPE_High skill tech staffをOCCUPATION_TYPE_High_skill_tech_staff
train.rename(columns={'OCCUPATION_TYPE_High skill tech staff': 'OCCUPATION_TYPE_High_skill_tech_staff'}, inplace=True)
test.rename(columns={'OCCUPATION_TYPE_High skill tech staff': 'OCCUPATION_TYPE_High_skill_tech_staff'}, inplace=True)
# OCCUPATION_TYPE_IT staffをOCCUPATION_TYPE_IT_staffに変更
train.rename(columns={'OCCUPATION_TYPE_IT staff': 'OCCUPATION_TYPE_IT_staff'}, inplace=True)
test.rename(columns={'OCCUPATION_TYPE_IT staff': 'OCCUPATION_TYPE_IT_staff'}, inplace=True)
# OCCUPATION_TYPE_Low-skill LaborersをOCCUPATION_TYPE_Low_skill_Laborersに変更
train.rename(columns={'OCCUPATION_TYPE_Low-skill Laborers': 'OCCUPATION_TYPE_Low_skill_Laborers'}, inplace=True)
test.rename(columns={'OCCUPATION_TYPE_Low-skill Laborers': 'OCCUPATION_TYPE_Low_skill_Laborers'}, inplace=True)
# OCCUPATION_TYPE_Medicine staffをOCCUPATION_TYPE_Medicine_staffに変更
train.rename(columns={'OCCUPATION_TYPE_Medicine staff': 'OCCUPATION_TYPE_Medicine_staff'}, inplace=True)
test.rename(columns={'OCCUPATION_TYPE_Medicine staff': 'OCCUPATION_TYPE_Medicine_staff'}, inplace=True)
# OCCUPATION_TYPE_Private service staffをOCCUPATION_TYPE_Private_service_staffに変更
train.rename(columns={'OCCUPATION_TYPE_Private service staff': 'OCCUPATION_TYPE_Private_service_staff'}, inplace=True)
test.rename(columns={'OCCUPATION_TYPE_Private service staff': 'OCCUPATION_TYPE_Private_service_staff'}, inplace=True)
# OCCUPATION_TYPE_Realty_agentsに変更
train.rename(columns={'OCCUPATION_TYPE_Realty agents': 'OCCUPATION_TYPE_Realty_agents'}, inplace=True)
test.rename(columns={'OCCUPATION_TYPE_Realty agents': 'OCCUPATION_TYPE_Realty_agents'}, inplace=True)
# OCCUPATION_TYPE_Sales_staffに変更
train.rename(columns={'OCCUPATION_TYPE_Sales staff': 'OCCUPATION_TYPE_Sales_staff'}, inplace=True)
test.rename(columns={'OCCUPATION_TYPE_Sales staff': 'OCCUPATION_TYPE_Sales_staff'}, inplace=True)
# OCCUPATION_TYPE_Security_staffに変更
train.rename(columns={'OCCUPATION_TYPE_Security staff': 'OCCUPATION_TYPE_Security_staff'}, inplace=True)
test.rename(columns={'OCCUPATION_TYPE_Security staff': 'OCCUPATION_TYPE_Security_staff'}, inplace=True)
# OCCUPATION_TYPE_Security_staffに変更
train.rename(columns={'OCCUPATION_TYPE_Security staff': 'OCCUPATION_TYPE_Security_staff'}, inplace=True)
test.rename(columns={'OCCUPATION_TYPE_Security staff': 'OCCUPATION_TYPE_Security_staff'}, inplace=True)
# OCCUPATION_TYPE_Waiters/barmen_staffに変更
train.rename(columns={'OCCUPATION_TYPE_Waiters/barmen staff': 'OCCUPATION_TYPE_Waiters/barmen_staff'}, inplace=True)
test.rename(columns={'OCCUPATION_TYPE_Waiters/barmen staff': 'OCCUPATION_TYPE_Waiters/barmen_staff'}, inplace=True)
# NAME_INCOME_TYPE_Commercial_associateに変更
train.rename(columns={'NAME_INCOME_TYPE_Commercial associate': 'NAME_INCOME_TYPE_Commercial_associate'}, inplace=True)
test.rename(columns={'NAME_INCOME_TYPE_Commercial associate': 'NAME_INCOME_TYPE_Commercial_associate'}, inplace=True)
# NAME_INCOME_TYPE_State_servantに変更
train.rename(columns={'NAME_INCOME_TYPE_State servant': 'NAME_INCOME_TYPE_State_servant'}, inplace=True)
test.rename(columns={'NAME_INCOME_TYPE_State servant': 'NAME_INCOME_TYPE_State_servant'}, inplace=True)
# NAME_EDUCATION_TYPE_Academic_degreeに変更
train.rename(columns={'NAME_EDUCATION_TYPE_Academic degree': 'NAME_EDUCATION_TYPE_Academic_degree'}, inplace=True)
test.rename(columns={'NAME_EDUCATION_TYPE_Academic degree': 'NAME_EDUCATION_TYPE_Academic_degree'}, inplace=True)
# NAME_EDUCATION_TYPE_Higher_educationに変更
train.rename(columns={'NAME_EDUCATION_TYPE_Higher education': 'NAME_EDUCATION_TYPE_Higher_education'}, inplace=True)
test.rename(columns={'NAME_EDUCATION_TYPE_Higher education': 'NAME_EDUCATION_TYPE_Higher_education'}, inplace=True)
# NAME_EDUCATION_TYPE_Incomplete_higherに変更
train.rename(columns={'NAME_EDUCATION_TYPE_Incomplete higher': 'NAME_EDUCATION_TYPE_Incomplete_higher'}, inplace=True)
test.rename(columns={'NAME_EDUCATION_TYPE_Incomplete higher': 'NAME_EDUCATION_TYPE_Incomplete_higher'}, inplace=True)
# NAME_EDUCATION_TYPE_Lower_secondaryに変更
train.rename(columns={'NAME_EDUCATION_TYPE_Lower secondary': 'NAME_EDUCATION_TYPE_Lower_secondary'}, inplace=True)
test.rename(columns={'NAME_EDUCATION_TYPE_Lower secondary': 'NAME_EDUCATION_TYPE_Lower_secondary'}, inplace=True)
# NAME_FAMILY_STATUS_Civil_marriageに変更
train.rename(columns={'NAME_FAMILY_STATUS_Civil marriage': 'NAME_FAMILY_STATUS_Civil_marriage'}, inplace=True)
test.rename(columns={'NAME_FAMILY_STATUS_Civil marriage': 'NAME_FAMILY_STATUS_Civil_marriage'}, inplace=True)
# NAME_HOUSING_TYPE_Co_op_apartmentに変更
train.rename(columns={'NAME_HOUSING_TYPE_Co-op apartment':'NAME_HOUSING_TYPE_Co_op_apartment'}, inplace = True)
test.rename(columns={'NAME_HOUSING_TYPE_Co-op apartment':'NAME_HOUSING_TYPE_Co_op_apartment'}, inplace = True)
# NAME_HOUSING_TYPE_Municipal_apartmentに変更
train.rename(columns={'NAME_HOUSING_TYPE_Municipal apartment':'NAME_HOUSING_TYPE_Municipal_apartment'}, inplace = True)
test.rename(columns={'NAME_HOUSING_TYPE_Municipal apartment':'NAME_HOUSING_TYPE_Municipal_apartment'}, inplace = True)
# NAME_HOUSING_TYPE_Office_apartmentに変更
train.rename(columns={'NAME_HOUSING_TYPE_Office apartment':'NAME_HOUSING_TYPE_Office_apartment'}, inplace = True)
test.rename(columns={'NAME_HOUSING_TYPE_Office apartment':'NAME_HOUSING_TYPE_Office_apartment'}, inplace = True)
# NAME_HOUSING_TYPE_Rented_apartmentに変更
train.rename(columns={'NAME_HOUSING_TYPE_Rented apartment':'NAME_HOUSING_TYPE_Rented_apartment'}, inplace = True)
test.rename(columns={'NAME_HOUSING_TYPE_Rented apartment':'NAME_HOUSING_TYPE_Rented_apartment'}, inplace = True)
# NAME_HOUSING_TYPE_With_parentsに変更
train.rename(columns={'NAME_HOUSING_TYPE_With parents':'NAME_HOUSING_TYPE_With_parents'}, inplace = True)
test.rename(columns={'NAME_HOUSING_TYPE_With parents':'NAME_HOUSING_TYPE_With_parents'}, inplace = True)
# NAME_TYPE_SUITE_Group_of_peopleに変更
train.rename(columns={'NAME_TYPE_SUITE_Group of people':'NAME_TYPE_SUITE_Group_of_people'}, inplace = True)
test.rename(columns={'NAME_TYPE_SUITE_Group of people':'NAME_TYPE_SUITE_Group_of_people'}, inplace = True)
# NAME_TYPE_SUITE_Spouse, partnerをNAME_TYPE_SUITE_Spouse_partnerに変更
train.rename(columns={'NAME_TYPE_SUITE_Spouse, partner':'NAME_TYPE_SUITE_Spouse_partner'}, inplace = True)
test.rename(columns={'NAME_TYPE_SUITE_Spouse, partner':'NAME_TYPE_SUITE_Spouse_partner'}, inplace = True)



def impute_missing_with_catboost_allow_missing(df, target_column, exclude_columns=None, n_features=10):
    
    if exclude_columns is None:
        exclude_columns = []

    # 欠損値がある行とない行を分割
    missing_rows = df[df[target_column].isnull()]
    non_missing_rows = df[df[target_column].notnull()]

    # 学習用データの準備（指定された列を除外）
    X_train_full = non_missing_rows.drop(columns=[target_column] + exclude_columns)
    y_train = non_missing_rows[target_column]

    # 特徴量選択（欠損値があっても動作可能）
    selector = SelectKBest(score_func=f_regression, k=n_features)
    X_train_selected = selector.fit_transform(X_train_full.fillna(-999), y_train)  # 欠損値を仮値で埋める（スコア計算用）
    selected_features = X_train_full.columns[selector.get_support()]

    # 欠損行に対しても同じ特徴量を適用
    X_missing = missing_rows[selected_features]

    # CatBoostで補完（欠損値は自動で扱われる）
    catboost_model = CatBoostRegressor(
        depth=4,
        verbose=0,
        loss_function='RMSE',
        random_state=42
    )
    catboost_model.fit(X_train_full[selected_features], y_train, cat_features=None)  # CatBoostは欠損値を含むデータで学習可能

    # 欠損値の予測と補完
    predicted_values = catboost_model.predict(X_missing)
    df.loc[missing_rows.index, target_column] = predicted_values

    return df


# EXT_SOURCE_2の欠損値をLightGBMで補完（説明変数に欠損値を許容）
train = impute_missing_with_catboost_allow_missing(
    train,
    target_column="EXT_SOURCE_2",
    exclude_columns=["TARGET"],  # TARGET列を除外
    n_features=50
)
test = impute_missing_with_catboost_allow_missing(
    test,
    target_column="EXT_SOURCE_2",
    n_features=50
)

# EXT_SOURCE_3の欠損値をLightGBMで補完（説明変数に欠損値を許容）
train = impute_missing_with_catboost_allow_missing(
    train,
    target_column="EXT_SOURCE_3",
    exclude_columns=["TARGET"],  # TARGET列を除外
    n_features=50
)
test = impute_missing_with_catboost_allow_missing(
    test,
    target_column="EXT_SOURCE_3",
    n_features=50
)

# EXT_SOURCE_1の欠損値をLightGBMで補完（説明変数に欠損値を許容）
train = impute_missing_with_catboost_allow_missing(
    train,
    target_column="EXT_SOURCE_1",
    exclude_columns=["TARGET"],  # TARGET列を除外
    n_features=50
)
test = impute_missing_with_catboost_allow_missing(
    test,
    target_column="EXT_SOURCE_1",
    n_features=50
)

# DAYS_BIRTH / REGION_RATING_CLIENTの特徴量を作成
train["DAYS_BIRTH / REGION_RATING_CLIENT"] = round(train["DAYS_BIRTH"] / train["REGION_RATING_CLIENT"], 3)
test["DAYS_BIRTH / REGION_RATING_CLIENT"] = round(test["DAYS_BIRTH"] / test["REGION_RATING_CLIENT"], 3)

# OBS_60_CNT_SOCIAL_CIRCLEを消去(確定12/1)
train.drop('OBS_60_CNT_SOCIAL_CIRCLE', axis=1, inplace=True)
test.drop('OBS_60_CNT_SOCIAL_CIRCLE', axis=1, inplace=True)

# FLAG_MOBILを消去
train.drop('FLAG_MOBIL', axis=1, inplace=True)
test.drop('FLAG_MOBIL', axis=1, inplace=True)

# DAYS_BIRTH + DAYS_ID_PUBLISHを作成
train["DAYS_BIRTH + DAYS_ID_PUBLISH"] = round(train["DAYS_BIRTH"] + train["DAYS_ID_PUBLISH"], 3)
test["DAYS_BIRTH + DAYS_ID_PUBLISH"] = round(test["DAYS_BIRTH"] + test["DAYS_ID_PUBLISH"], 3)
# DAYS_BIRTH + DAYS_ID_PUBLISHとTARGETの相関係数
train["DAYS_BIRTH + DAYS_ID_PUBLISH"].corr(train["TARGET"])

#13
# DAYS_ID_PUBLISH + DAYS_LAST_ PHONE_CHANGEとTARGETを作成
train["DAYS_ID_PUBLISH + DAYS_LAST_PHONE_CHANGE"] = round(train["DAYS_ID_PUBLISH"] + train["DAYS_LAST_PHONE_CHANGE"], 3)
test["DAYS_ID_PUBLISH + DAYS_LAST_PHONE_CHANGE"] = round(test["DAYS_ID_PUBLISH"] + test["DAYS_LAST_PHONE_CHANGE"], 3)
# DAYS_ID_PUBLISH + DAYS_LAST_ PHONE_CHANGEとTARGETの相関係数
train["DAYS_ID_PUBLISH + DAYS_LAST_PHONE_CHANGE"].corr(train["TARGET"])


# AMT_CREDIT / AMT_INCOME_TOTALを作成(確定)
train["AMT_CREDIT / AMT_INCOME_TOTAL"] = round(train["AMT_CREDIT"] / train["AMT_INCOME_TOTAL"], 3)
test["AMT_CREDIT / AMT_INCOME_TOTAL"] = round(test["AMT_CREDIT"] / test["AMT_INCOME_TOTAL"], 3)

# AMT_ANNUITY / AMT_INCOME_TOTALを作成
train["AMT_ANNUITY / AMT_INCOME_TOTAL"] = round(train["AMT_ANNUITY"] / train["AMT_INCOME_TOTAL"], 3)
test["AMT_ANNUITY / AMT_INCOME_TOTAL"] = round(test["AMT_ANNUITY"] / test["AMT_INCOME_TOTAL"], 3)

# CREDIT_TERM: AMT_CREDIT / AMT_ANNUITYを作成(確定)
train["CREDIT_TERM"] = round(train["AMT_CREDIT"] / train["AMT_ANNUITY"], 3)
test["CREDIT_TERM"] = round(test["AMT_CREDIT"] / test["AMT_ANNUITY"], 3)

# GOODS_PRICE_CREDIT_RATIO: AMT_GOODS_PRICE ÷ AMT_CREDITを作成(確定)
train["GOODS_PRICE_CREDIT_RATIO"] = round(train["AMT_GOODS_PRICE"] / train["AMT_CREDIT"], 3)
test["GOODS_PRICE_CREDIT_RATIO"] = round(test["AMT_GOODS_PRICE"] / test["AMT_CREDIT"], 3)

# CREDIT_GOODS_DIFF: AMT_CREDIT - AMT_GOODS_PRICEを作成(確定)
train["CREDIT_GOODS_DIFF"] = round(train["AMT_CREDIT"] - train["AMT_GOODS_PRICE"])
test["CREDIT_GOODS_DIFF"] = round(test["AMT_CREDIT"] - test["AMT_GOODS_PRICE"])

# CHILDREN_RATIO: CNT_CHILDREN ÷ (CNT_CHILDREN + 1)を作成(確定)
train["CHILDREN_RATIO"] = round(train["CNT_CHILDREN"] / (train["CNT_CHILDREN"] + 1), 3)
test["CHILDREN_RATIO"] = round(test["CNT_CHILDREN"] / (test["CNT_CHILDREN"] + 1), 3)




# 重要特徴量を基にした新しい特徴量の作成
#6
train["EXT_SOURCE_1 / EXT_SOURCE_2"] = train["EXT_SOURCE_1"] / (train["EXT_SOURCE_2"] + 1e-6)
test["EXT_SOURCE_1 / EXT_SOURCE_2"] = test["EXT_SOURCE_1"] / (test["EXT_SOURCE_2"] + 1e-6)

# 1
# クラスタリング
clustering_features_train = train[["CREDIT_TERM", "EXT_SOURCE_3", "EXT_SOURCE_2"]].fillna(0)
clustering_features_test = test[["CREDIT_TERM", "EXT_SOURCE_3", "EXT_SOURCE_2"]].fillna(0)

kmeans = KMeans(n_clusters=5, random_state=42)
train["Cluster"] = kmeans.fit_predict(clustering_features_train)
test["Cluster"] = kmeans.predict(clustering_features_test)

# クラスタとの組み合わせ特徴量
#train["Cluster * EXT_SOURCE_3"] = train["Cluster"] * train["EXT_SOURCE_3"]
train["Cluster / CREDIT_TERM"] = train["Cluster"] / (train["CREDIT_TERM"] + 1e-6)

#test["Cluster * EXT_SOURCE_3"] = test["Cluster"] * test["EXT_SOURCE_3"]
test["Cluster / CREDIT_TERM"] = test["Cluster"] / (test["CREDIT_TERM"] + 1e-6)


# 時間関連の特徴量
#5
train["DAYS_EMPLOYED - DAYS_REGISTRATION"] = train["DAYS_EMPLOYED"] - train["DAYS_REGISTRATION"]
test["DAYS_EMPLOYED - DAYS_REGISTRATION"] = test["DAYS_EMPLOYED"] - test["DAYS_REGISTRATION"]


pd.set_option('display.max_info_columns', 300)



# ライブラリの読み込み
# 目的変数と説明変数に分割
X = train.drop("TARGET", axis=1).values
y = train["TARGET"].values
X_test = test.values

# 訓練データと評価データに分割
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


# KFold以外でも、差がないようにうまく分ける。



def lgbm_cross_validation(X, y, n_splits=10, params=None):
    if params is None:
        params = {
           'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'learning_rate': 0.1,
            'max_depth': 7,
            'num_leaves': 31,
            'random_state': 42,
            'verbose': -1
        }

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)
    auc_scores = []

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        print(f"Fold {fold + 1}/{n_splits}")

        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        # データセット作成
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

        # モデル学習
        model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, valid_data]
        )

        # 予測と評価
        y_pred = model.predict(X_valid, num_iteration=model.best_iteration)
        auc = roc_auc_score(y_valid, y_pred)
        auc_scores.append(auc)
        print(f"Fold {fold + 1} AUC: {auc:.4f}")

    mean_auc = sum(auc_scores) / n_splits
    print(f"Mean AUC: {mean_auc:.4f}")
    return mean_auc, auc_scores

'''
# 交差検証
mean_auc, auc_scores = lgbm_cross_validation(X, y, n_splits=10)
'''
from lightgbm import LGBMClassifier

param = {
           'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'learning_rate': 0.1,
            'max_depth': 7,
            'num_leaves': 31,
            'random_state': 42,
            'verbose': -1
        }

model = LGBMClassifier(**param)
model.fit(X, y)

pred = model.predict_proba(X_test)[:, 1]

sample_sub["TARGET"] = pred



# Excelファイルの保存先と名前を指定
output_path = "C:/Users/81906/Desktop/submission_extra.csv"

# DataFrameをExcelファイルに保存
sample_sub.to_csv(output_path, index=False)

print(f"Excelファイルが保存されました: {output_path}")



'''import optuna # type: ignore
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Optunaの目的関数
def objective(trial):
    # ハイパーパラメータの提案
    param = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "num_leaves": trial.suggest_int("num_leaves", 10, 150),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 10),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10),
        "random_state": 42,
        "n_estimators": 500,  # 固定値、必要に応じて最適化
        "verbose": -1
    }

    # クロスバリデーション
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in kf.split(X_train, y_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        model = LGBMClassifier(**param)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
        preds = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, preds)
        scores.append(score)

    return np.mean(scores)

# Optunaによる最適化
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# 最適なハイパーパラメータの取得
best_params = study.best_params
print("Best Parameters:", best_params)

# 最適なハイパーパラメータでモデルを再学習
final_model = LGBMClassifier(**best_params, random_state=42)
final_model.fit(X_train, y_train)

# 検証データでの予測
final_preds = final_model.predict_proba(X_valid)[:, 1]

# AUCスコアを計算
final_auc = roc_auc_score(y_valid, final_preds)
print(f"Final Model AUC: {final_auc:.4f}")'''





