from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

STUDENT_CELLS = open('student/student-mat.csv', 'r')


# process_file splits the file into features and reviews
def process_file(file):
    features = []
    labels = []
    for line in file:
        s = line.split(';')
        # features are study time, grade 1, grade 2
        features.append([int(s[13].replace('"', '')), int(s[31].replace('"', ''))])
        labels.append(s[32].replace('"', ''))
    return features, labels


# train_reg_model trains linear regression model and displays
# metrics in found in cross validation.
def train_reg_model(features, labels):
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.3)
    reg = linear_model.LinearRegression()
    reg.fit(x_train, y_train)

    # metrics / display
    y_predict = reg.predict(x_test)
    print('Coefficients: \n', reg.coef_)
    print(f'r2: {reg.score(x_test, y_test)} \n')
    print(f'mse: {mean_squared_error(y_test, y_predict)}')

    return reg


student_X, student_y = process_file(STUDENT_CELLS)
lin_reg = train_reg_model(student_X, student_y)



