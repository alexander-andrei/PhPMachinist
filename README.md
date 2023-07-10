1. Linear Regression: one of the simplest and most interpretable models. It assumes a linear relationship between the dependent variable and the independent variables. The model estimates the coefficients for each independent variable to minimize the sum of squared errors.

```php
use PhpMachinist\Models\LinearRegression;

$regression = new LinearRegression();
$ages = [18, 25, 32, 45, 60]; // array of ages
$heights = [160, 165, 170, 175, 180]; // array of heights
$regression->train($ages, $heights);

$testingData = [35, 10, 14];
$predictions = $regression->predict($testingData);
/**
* Prediction for age = 35: 169.53488372093
* Prediction for age = 10: 157.90697674419
* Prediction for age = 14: 159.76744186047
*/
```

2. Logistic Regression: an extension of linear regression used for binary classification problems. It models the probability of a binary outcome using a logistic function. It is relatively straightforward and widely used for binary classification tasks.

```php
// age and height
$peopleByAgeAndHeight = [
    [18, 165],
    [25, 170],
    [32, 175],
    [45, 180],
    [60, 185],
    [20, 160],
    [28, 168],
    [35, 172],
    [50, 178],
    [65, 190]
];

/**
 * 0 -> female
 * 1 -> male
 */
$peopleBySex = [0, 0, 0, 1, 1, 0, 0, 1, 1, 1];

$lr = new LogisticRegression();
$lr->train($peopleByAgeAndHeight, $peopleBySex);

$weights = $lr->getWeights();
$intercept = $lr->getIntercept();

echo "Learned Weights: " . implode(", ", $weights) . "\n";
echo "Intercept: $intercept\n";

$newSample = [35, 170];
$prediction = $lr->predict([$newSample])[0]; // Prediction: Female
```

3. Multinominal Naive Bayes: a probabilistic classification algorithm that is based on Bayes' theorem. It is suitable for classifying text or categorical data where features represent word counts or frequency of occurrence. Works very well with limited training data.

```php
// Dataset containing features for playing tennis
$dataset = [
    [1, 'Sunny', 'Hot', 'High', 'Weak'],
    [2, 'Sunny', 'Hot', 'High', 'Strong'],
    [3, 'Overcast', 'Hot', 'High', 'Weak'],
    [4, 'Rain', 'Mild', 'High', 'Weak'],
    [5, 'Rain', 'Cool', 'Normal', 'Weak'],
    [6, 'Rain', 'Cool', 'Normal', 'Strong'],
    [7, 'Overcast', 'Cool', 'Normal', 'Strong'],
    [8, 'Sunny', 'Mild', 'High', 'Weak'],
    [9, 'Sunny', 'Cool', 'Normal', 'Weak'],
    [10, 'Rain', 'Mild', 'Normal', 'Weak'],
    [11, 'Sunny', 'Mild', 'Normal', 'Strong'],
    [12, 'Overcast', 'Mild', 'High', 'Strong'],
    [13, 'Overcast', 'Hot', 'Normal', 'Weak'],
    [14, 'Rain', 'Mild', 'High', 'Strong']
];

// Labels for each feature to determine if we play tennis or not for each condition
$labels = [
    'No',
    'No',
    'Yes',
    'Yes',
    'Yes',
    'No',
    'Yes',
    'No',
    'Yes',
    'Yes',
    'Yes',
    'Yes',
    'Yes',
    'No'
];

$nbClassifier = new MultinomialNaiveBayes();
$nbClassifier->train($dataset, $labels);

/**
* Based on the features we want to predict if it is 
 * ok for us to play tennis today.
 */
$testFeatures = [5, 'Sunny', 'Cool', 'High', 'Weak'];
$predictedClass = $nbClassifier->predict($testFeatures); // Prediction: no
```

4. Decision Trees (TBD)

5. k-Nearest Neighbors (k-NN): k-NN is an instance-based algorithm that classifies new observations based on the similarity to the k nearest neighbors in the training data. It is simple to understand and implement, but can be computationally intensive for large datasets.

```php
// training data representing features of an F1 tyre and the associated class
$trainingData = [
    ['features' => [6.0, 2.2, 4.0, 1.0], 'class' => 'soft'],
    ['features' => [5.7, 3.0, 4.2, 1.2], 'class' => 'soft'],
    ['features' => [5.7, 2.9, 4.2, 1.3], 'class' => 'soft'],
    ['features' => [4.9, 3.1, 1.5, 0.1], 'class' => 'medium'],
    ['features' => [5.4, 3.7, 1.5, 0.2], 'class' => 'medium'],
    ['features' => [5.0, 3.4, 1.6, 0.4], 'class' => 'medium'],
    ['features' => [6.7, 3.1, 5.6, 2.4], 'class' => 'hard'],
    ['features' => [6.3, 2.5, 5.0, 1.9], 'class' => 'hard'],
    ['features' => [6.5, 3.0, 5.2, 2.0], 'class' => 'hard'],
];

$knn = new KNearestNeighbour(3);
$knn->train($trainingData);

$sample1 = [6.1, 2.8, 4.7, 1.2];
$prediction1 = $knn->predict($sample1); // soft tyre

$sample2 = [5.3, 3.2, 1.3, 0.4];
$prediction2 = $knn->predict($sample2); // medium tyre
```

6. Support Vector Machines (SVM): a versatile and powerful model used for both classification and regression tasks. It finds an optimal hyperplane that separates data points of different classes or predicts the value of a continuous variable. SVM can handle complex relationships but may require careful tuning of hyperparameters.

```php
// age/height
$samples = [
    [18, 160],
    [25, 165],
    [32, 170],
    [45, 175],
    [60, 180]
];
$labels = [0, 0, 1, 1, 1]; // 0 represents short, 1 represents tall

// Create and train the SVM classifier
$classifier = new SVM('polynomial');  // linear and rbf kernel can also be used
$classifier->train($samples, $labels);

$newSample = [32, 0];
$predictedLabel = $classifier->predict($newSample);
```

7. Feed forward neural network: a type of artificial neural network where information flows in one direction, from the input layer through one or more hidden layers to the output layer. It is called "feedforward" because there are no feedback connections or loops within the network.
```php
$layerSizes = [2, 3, 1]; // Number of neurons in each layer
$network = new NeuralNetwork($layerSizes);

$inputs = [25, 170]; // Input values

$outputs = $network->forward($inputs);// Result: 0.65458049013959 
```