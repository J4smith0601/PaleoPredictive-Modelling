# PaleoPredictive-Modelling ReadME

## Title:

**PaleoPredictive Modelling: Leveraging Machine Learning for Fossil Classification**

---

**Project Overview: PaleoPredictive Modelling**

**Introduction:**
Palaeontology encounters significant challenges in studying fossils due to the incomplete and biased nature of fossil records. Fossils typically consist only of skeletal remains, making it difficult to reconstruct full biological and ecological profiles. Destructive techniques are often required for detailed internal analysis, and the fossil record is limited by preservation biases and rarity.

**Project Aims:**
The "PaleoPredictive Modelling" project aimed to evaluate the applicability and effectiveness of machine learning techniques in palaeontological research. The primary goal was to develop predictive models for classifying fossil species using various data inputs, including skeletal measurements and discovery characteristics, to offer new insights into the ancient past.

**Model Performance:**

- **Decision Tree:** Configured with a maximum depth of 15, this model demonstrated strong performance with an accuracy of 0.83 across both training and testing phases. It managed classification tasks effectively but faced challenges with class imbalance, suggesting that stratified sampling or resampling methods might improve robustness.
  
- **Pruned Decision Tree (DT2):** With its depth reduced from 15 to 5, this model aimed to address overfitting. Although it improved the learning curve, it suffered from underfitting, resulting in a decline in performance metrics.

- **Random Forest:** This model, with 100 trees and a maximum depth of 15, achieved the highest performance, with a training accuracy of 0.97 and a testing accuracy of 0.87. It reduced variance and mitigated overfitting but required further refinement to address data imbalance and enhance performance.

**Advanced Techniques and Results:**

- **Synthetic Data Generation:** Techniques such as SMOTE and ensemble bagging were employed to address sample size imbalances. New models, including the Bagging Classifier with K-Nearest Neighbours (KNN) and an extended Random Forest (RF2), showed perfect training metrics but exhibited severe overfitting.

- **External Validation:** Validation with a withheld data subset revealed varied performance. The Decision Tree (DT1) achieved 33.33% accuracy, while the Random Forest (RF1) achieved 66.67%. Extended Random Forest (RF2) and K-Nearest Neighbours (KNN) had significantly lower accuracies, highlighting overfitting and poor generalisation.

**Comparison with Traditional Methods:**
Traditional palaeontological methods achieved a baseline accuracy of 77.21%. Although machine learning models demonstrated potential, they fell short of this benchmark, indicating challenges in achieving robust performance.

**Contributions and Future Directions:**

- **Integration of Machine Learning:** Introduces advanced computational methods to palaeontology, revealing complex patterns in fossil data.
- **Framework Development:** Establishes a comprehensive classification framework.
- **Benchmarking:** Provides a comparative evaluation of machine learning models against traditional methods.
- **Future Research:** Identifies key areas for improvement, including sample size expansion, data collection, model refinement, and exploration of advanced techniques.

Overall, while the study highlights the potential of machine learning to enhance palaeontological research, it also underscores the need for further refinement, addressing class imbalance, and exploring alternative algorithms to improve predictive accuracy and handle the limitations of fossil data.

---

## Technologies Used

This project employs a range of technologies and libraries for data analysis, visualisation, and machine learning:

- **NumPy**: Fundamental package for scientific computing and numerical operations.
- **Pandas**: Applied for efficient data manipulation and analysis.
- **Matplotlib**: Utilised for creating static, animated, and interactive visualisations.
- **Seaborn**: Provides advanced statistical data visualisation.
- **Plotly Express**: Enables high-level interactive visualisations.
- **sklearn.tree.plot_tree**: Visualises decision tree models.
- **Geopy**:
  - **GeocoderTimedOut, GeocoderServiceError**: Handles exceptions during geocoding.
  - **Nominatim**: Geocodes coordinates to impute missing location data.
- **Geopandas**: Extends Pandas for spatial data operations.
- **SciPy**:
  - **kurtosis, skew, shapiro, norm**: Performs statistical assessments and normality tests.
  - **chi2_contingency**: Evaluates independence in contingency tables.
  - **boxcox**: Applies Box-Cox transformations to stabilise variance.
- **sklearn.preprocessing**:
  - **LabelEncoder**: Converts categorical variables into numerical labels.
  - **OneHotEncoder**: Creates binary vectors for categorical features.
  - **MinMaxScaler**: Standardises feature scores to a [0, 1] range.
  - **StandardScaler**: Normalises numeric features to mean 0 and std dev 1.
- **sklearn.feature_selection**:
  - **SelectFromModel**: Selects features based on model importance.
  - **chi2, SelectKBest**: Uses chi-squared test and selects top k features.
- **sklearn.neighbors**: Implements k-Nearest Neighbours classifier.
- **sklearn.tree**: Provides Decision Tree classifier for various tasks.
- **sklearn.ensemble**:
  - **RandomForestClassifier**: Robust ensemble learning with Random Forest.
  - **BaggingClassifier**: Implements Bagging for improved performance.
- **sklearn.model_selection**:
  - **cross_val_score**: Evaluates model performance using cross-validation.
  - **train_test_split, learning_curve**: Splits data and plots learning curves.
  - **LeaveOneOut**: Performs Leave-One-Out Cross-Validation (LOOCV).
  - **GridSearchCV, RandomizedSearchCV**: Conducts hyperparameter tuning.
- **sklearn.metrics**:
  - **accuracy_score, classification_report, precision_score, mean_squared_error, confusion_matrix, recall_score, f1_score, auc, roc_auc_score, roc_curve**: Metrics for evaluating model performance.
- **imblearn.over_sampling**:
  - **SMOTE**: Balances class distribution through synthetic oversampling.
- **imblearn.under_sampling**:
  - **RandomUnderSampler**: Reduces majority class size to address imbalance.
- **Statsmodels**:
  - **api**: Conducts power analysis for sample size determination.
- **collections.Counter**: Counts hashable objects in an iterable.
- **sklearn.preprocessing.label_binarize**: Binarises labels for multiclass classification.
- **Python**: The primary programming language for data analysis and processing.
- **Jupyter Notebook**: For interactive analysis and documenting the process.
- **VS Code**: The IDE used for coding and managing the project.
- **Git**: For version control and tracking changes.
- **GitHub**: Hosts the repository and supports collaboration.

---

## Installation Instructions

To run this project locally, follow these steps:

**1. Clone the Repository:**
Open a terminal or command prompt and run the following command to clone the repository:
git clone https://github.com/J4smith0601/PaleoPredictive-Modelling.git

**2. Navigate to the Project Directory:**
Change your directory to the project folder:
cd [PaleoPredictive-Modelling]

**3. Install Dependencies:**
Ensure you have Python installed on your system. Then, install the required libraries using pip.
numpy
pandas
matplotlib
seaborn
plotly
scipy
geopy
geopandas
scikit-learn
imblearn
statsmodels

**4. Set Up Data:**
	-	Download the associated data files from the dataset provided or follow internal instructions using the Paleobiology database (https://paleobiodb.org/#/).
	-	Store the data files with the notebook. You may need to update file paths within the notebook to match where you have saved the files.

**5. Verify Installation:**
To verify that the dependencies are correctly installed, you can run a Python script or open a Jupyter Notebook and import the libraries to ensure they are available.

**6. Run the Jupyter Notebook:**
- Launch Jupyter Notebook and run all cells to execute the analysis.

## Usage

This project demonstrates the Machine Learning and Modelling capabilities acquired during the Data Science & AI course. It is designed to showcase skills in data manipulation, statistical analysis, and visualization, with a focus on applying machine learning techniques to real-world problems. The project aims to highlight expertise to potential stakeholders, future employers, and other interested parties.

---

## Results and Analysis 

### Key Findings from Machine Learning Analysis

The project, titled **PaleoPredictive Modelling: Leveraging Machine Learning for Fossil Species Classification**, aimed to evaluate the applicability and effectiveness of machine learning techniques in palaeontological research. The primary goal was to develop predictive models for classifying fossil species using various data inputs, including skeletal measurements and discovery characteristics.

**Model Performance:**

- **Decision Tree:** Configured with a maximum depth of 15 and featuring geographical and temporal data, the Decision Tree model achieved an accuracy of 0.83 across both training and testing phases. The model demonstrated good generalisation capabilities, as evidenced by consistent precision, recall, and F1 scores. However, class imbalance adversely affected its performance, suggesting that stratified sampling or resampling methods could improve evaluation robustness.

- **Pruned Decision Tree (DT2):** With the depth reduced from 15 to 5, DT2 aimed to address overfitting. While this adjustment improved the learning curve by narrowing the gap between training and cross-validation scores, it resulted in significant declines in performance metrics, indicating underfitting. This highlighted the challenge of balancing model complexity with generalisation.

- **Random Forest:** With 100 trees and a maximum depth of 15, the Random Forest model exhibited the highest performance among the models, achieving a training accuracy of 0.97 and a testing accuracy of 0.87. The ensemble approach effectively reduced variance and mitigated overfitting. Nonetheless, further data imbalance handling and parameter refinement could enhance its performance.

**Advanced Techniques and Additional Models:**

To tackle issues related to sample size imbalances and class distribution, advanced synthetic data generation techniques were employed, including the Synthetic Minority Over-sampling Technique (SMOTE), undersampling, and ensemble bagging. Two additional models were developed using this enhanced data:

- **Bagging Classifier with K-Nearest Neighbours (KNN):** Achieved perfect training metrics but demonstrated severe overfitting with significant discrepancies between training and cross-validation scores, indicating excessive sensitivity to the training data.

- **Extended Random Forest (RF2):** Also showed perfect training metrics but suffered from severe overfitting, with substantial gaps between training and cross-validation scores.

**External Validation:**

External validation using a deliberately withheld subset of data revealed varied performance outcomes:
- **Decision Tree Model (DT1):** Achieved an accuracy of 33.33%, reflecting limited generalisation capabilities.
- **Pruned Decision Tree (DT2):** Not assessed due to prior performance issues.
- **Random Forest Model (RF1):** Demonstrated a better accuracy of 66.67%, though still below the baseline.
- **Extended Random Forest (RF2) and K-Nearest Neighbours (KNN) Models:** Showed significantly lower accuracies of 16.67% and 0.00%, respectively, highlighting severe overfitting and poor generalisation.

In comparison, traditional palaeontological methods achieved a baseline accuracy of 77.21%, indicating effective species identification. Although the machine learning models showed promise, they fell short of this benchmark, underscoring ongoing challenges in achieving robust performance. The results emphasise the need for further model refinement, exploration of alternative algorithms, and advanced techniques to improve predictive accuracy and address class imbalance in palaeontological research.

**Conclusion:**

The study highlights the potential of machine learning to enhance palaeontological research but also underscores the challenges in achieving robust and generalisable models. While some models showed promising results, further refinement and continued efforts are necessary to address the limitations and uncertainties associated with fossil data.

---

## License

This project is not licensed under any specific license. You are free to use, modify, and distribute this project as you see fit. However, please note that this project is intended for personal use and portfolio showcasing purposes. If you share or distribute this project, please ensure that any derivative works or modifications adhere to the same principles of personal use and respect for the original work.

---

## Acknowledgements

Here’s an updated acknowledgment section for your project 
- The dataset used in this project was sourced from the Paleobiology Database (https://paleobiodb.org/#/). This extensive fossil data provided the foundation for the machine learning models developed in this project. Appreciation is extended to the contributors of the Paleobiology Database for their invaluable resource.
- Special thanks to Laks Balasubramanian and the Data Science & AI course team at the Institute of Data and Auckland University of Technology for their guidance and support throughout this project.
- Acknowledgment is given to the Python libraries used, including NumPy, Pandas, Matplotlib, Seaborn, Plotly, SciPy, Geopy, Geopandas, Scikit-learn, Imbalanced-learn, and Statsmodels, which were essential for data manipulation, analysis, and visualization.
- No external funding or grants were received for this project.

---

### Data Source

The original dataset used in this project was obtained from the Paleobiology Database (https://paleobiodb.org/#/). This extensive resource provided comprehensive fossil data, which served as the foundation for the machine learning models developed in this project.

For more information about the data, including its structure and additional details, you can visit the Paleobiology Database repository.

---

## Contact Information

For feedback, networking, or inquiries related to this project, please feel free to reach out:

- LinkedIn: https://www.linkedin.com/in/jasmith0601/
- Email: James.smith@andersmith.co.nz

You can connect with me via LinkedIn or email to discuss the project, explore collaboration opportunities, or provide feedback.
