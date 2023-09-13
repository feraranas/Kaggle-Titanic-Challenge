import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
import pickle
import joblib
from sklearn.linear_model import LogisticRegression


# ////////////////////////
# TITLE & TEAM INFO
# ////////////////////////
st.title('Titanic Survivors analysis')
team = pd.DataFrame({
     'Alumnos': [
         'Mauricio Juárez Sánchez',
         'Alfredo Jeong Hyun Park',
         'Fernando Alfonso Arana Salas',
         'Miguel Ángel Bustamante Pérez']
     })
st.write(team)

# load data




