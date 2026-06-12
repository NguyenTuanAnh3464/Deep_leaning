"""Nhãn và giá trị mẫu dùng trong project."""

CIFAR10_LABELS = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

MNIST_LABELS = [str(i) for i in range(10)]

CATDOG_LABELS = ["cat", "dog"]

ADULT_NUMERIC_FEATURES = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

ADULT_CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

ADULT_FEATURES = ADULT_NUMERIC_FEATURES + ADULT_CATEGORICAL_FEATURES

ADULT_DEFAULT_OPTIONS = {
    "workclass": ["Private", "Self-emp-not-inc", "Local-gov", "State-gov", "Federal-gov"],
    "education": ["HS-grad", "Some-college", "Bachelors", "Masters", "Assoc-voc"],
    "marital-status": ["Never-married", "Married-civ-spouse", "Divorced", "Separated"],
    "occupation": ["Adm-clerical", "Craft-repair", "Exec-managerial", "Prof-specialty", "Sales"],
    "relationship": ["Not-in-family", "Husband", "Own-child", "Unmarried", "Wife"],
    "race": ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"],
    "sex": ["Male", "Female"],
    "native-country": ["United-States", "Vietnam", "Philippines", "Mexico", "Canada"],
}

ADULT_LABELS = ["<=50K", ">50K"]

CAR_FEATURES = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]

CAR_OPTIONS = {
    "buying": ["vhigh", "high", "med", "low"],
    "maint": ["vhigh", "high", "med", "low"],
    "doors": ["2", "3", "4", "5more"],
    "persons": ["2", "4", "more"],
    "lug_boot": ["small", "med", "big"],
    "safety": ["low", "med", "high"],
}

CAR_LABELS = ["unacc", "acc", "good", "vgood"]
