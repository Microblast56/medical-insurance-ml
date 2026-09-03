import pandas as pd


MODEL_FEATURES = [
    "age",
    "sex",
    "bmi",
    "children",
    "smoker",
    "region_northwest",
    "region_southeast",
    "region_southwest"
]


def preprocess_input(age, sex, bmi, children, smoker, region):
    sex_encoded = 1 if sex == "male" else 0
    smoker_encoded = 1 if smoker == "yes" else 0

    input_data = pd.DataFrame({
        "age": [age],
        "sex": [sex_encoded],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker_encoded],
        "region_northwest": [1 if region == "northwest" else 0],
        "region_southeast": [1 if region == "southeast" else 0],
        "region_southwest": [1 if region == "southwest" else 0]
    })

    return input_data[MODEL_FEATURES]