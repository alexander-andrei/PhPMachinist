<?php

namespace PhpMachinist\Models;

/**
 * Linear regression is a statistical modeling technique used to understand the relationship
 * between a dependent variable and one or more independent variables. It assumes a linear
 * relationship and aims to find the best-fit line that minimizes the difference between
 * the predicted values and the actual values. The resulting model can be used to predict
 * the values of the dependent variable based on the values of the independent variables.
 *
 * It is best used in situations where there is a linear relationship between the dependent
 * variable and one or more independent variables.
 *
 * Formula: Y = b + mX + ε
 * Y is the dependent variable (the variable being predicted or explained).
 * X is the independent variable (the predictor variable).
 * b is the intercept term (the value of Y when X is zero).
 * m is the slope (the rate of change of Y with respect to X).
 * ε is the error term (the residual or unexplained variation in Y not accounted for by the model).
 */
class LinearRegression
{
    private float $slope;
    private float $intercept;

    /**
     * @param array $xData => an array of ages ([18, 25, 32, 45, 60])
     * @param array $yData => an array of heights ([160, 165, 170, 175, 180])
     */
    public function train(array $xData, array $yData): void
    {
        $xDataSize = count($xData);

        /**
         * Calculating the mean
         *
         * Mean => a mean is the sum of all numbers in an array
         * divided by how many numbers are in that array
         */
        $meanX = array_sum($xData) / $xDataSize; // 180 / 5 => 36
        $meanY = array_sum($yData) / $xDataSize; // 850 /5 => 170

        // Calculate the slope (m)

        /**
         * $numerator => refers to the sum of the product of the deviations of the independent variable (X)
         * from its mean and the corresponding deviations of the dependent variable (Y) from its mean.
         * It represents the co-variation between X and Y.
         *
         * $denominator => refers to the sum of the squared deviations of the independent variable (X) from its mean.
         * It represents the variance of X.
         */
        $numerator = 0;
        $denominator = 0;

        for ($i = 0; $i < $xDataSize; $i++) {
            /**
             * Calculations for $numerator would be:
             * $numerator += (18 -36) * (160 - 170) => 180
             * $numerator += (25 -36) * (165 - 170) => 235
             * $numerator += (32 -36) * (170 - 170) => 235
             * $numerator += (45 -36) * (175 - 170) => 280
             * $numerator += (60 -36) * (180 - 170) => 520
             *
             * $numerator = 520
             */
            $numerator += ($xData[$i] - $meanX) * ($yData[$i] - $meanY);

            /**
             * Calculations for $denominator would be:
             * $denominator += (18 - 36) ** 2 => 324
             * $denominator += (25 - 36) ** 2 => 445
             * $denominator += (32 - 36) ** 2 => 461
             * $denominator += (45 - 36) ** 2 => 542
             * $denominator += (60 - 36) ** 2 => 1118
             *
             * $denominator = 1118
             */
            $denominator += ($xData[$i] - $meanX) ** 2;
        }

        /**
         * The slope (m) is calculated (m = numerator / denominator)
         *
         * $slope = 520 / 1118 => 0.4651
         */
        $slope = $numerator / $denominator;

        /**
         * The intercept (b) is the point where the regression line intersects the y-axis.
         * In linear regression Y = b + mX, thus to calculate b we need to do b = Y - mX
         *
         * b = 170 - 0.4651 * 36 => 153.25
         */
        $intercept = $meanY - $slope * $meanX;

        $this->slope = $slope;
        $this->intercept = $intercept;
    }

    public function predict(array $xData): array
    {
        $predictions = [];
        foreach ($xData as $x) {
            /**
             * Calculation would be:
             *
             * $prediction = m * $x * b => 0.4651 * $x + 153.25
             */
            $predictions[] = $this->slope * $x + $this->intercept;
        }
        return $predictions;
    }
}



