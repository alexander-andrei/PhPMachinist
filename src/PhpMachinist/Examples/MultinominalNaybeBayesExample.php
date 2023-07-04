<?php

use PhpMachinist\Models\MultinomialNaiveBayes;

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

$testFeatures = [5, 'Sunny', 'Cool', 'High', 'Weak'];
$predictedClass = $nbClassifier->predict($testFeatures);

echo "Predicted class: $predictedClass\n";
