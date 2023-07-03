<?php

namespace PhpMachinist\NeuralNetworks\FeedForward;

class Neuron {
    private array $weights;
    private int $bias;

    public function __construct($numInputs) {
        $this->weights = [];

        for ($i = 0; $i < $numInputs; $i++) {
            /**
             * Weights are the parameters of the neural network that determine the strength
             * of the connections between neurons. Each input feature is associated with a
             * weight in a neuron, indicating the importance or influence of that feature on
             * the neuron's output. The weights are adjusted during the training process to
             * optimize the network's performance. By adjusting the weights, the neural network
             * can learn to assign different levels of importance to different input features
             * allowing it to capture complex patterns and make accurate predictions.
             *
             * The purpose of initializing the weights with random values is to introduce
             * variability and ensure that the neural network starts with different initial
             * weights, which can help prevent the network from getting stuck in suboptimal
             * solutions during training.
             *
             * This is just a simple example and more advanced weight initialization techniques
             * are often used, such as Xavier or He initialization, which take into account the
             * number of inputs and outputs to ensure better convergence properties.
             *
             * $this->weights[] => a floating point number between -1 and 1 for each neuron
             */
            $this->weights[] = rand(-100, 100) / 100;
        }

        /**
         * The bias is an additional parameter in each neuron that allows for the introduction
         * of an offset or intercept term. It represents the neuron's inherent bias towards
         * firing or not firing, regardless of the input values. The bias term helps the neural
         * network to model situations where the input features alone may not be sufficient to
         * make accurate predictions. It provides flexibility to the model by allowing it to
         * shift the decision boundary or activation threshold. Like weights, the bias term is
         * adjusted during training to improve the network's performance.
         *
         * $this->bias => a floating point number between -1 and 1
         */
        $this->bias = rand(-100, 100) / 100;
    }

    public function activate($inputs) {
        $sum = $this->bias;

        /**
         * To calculate the weighted sum, we multiply each input value by its corresponding
         * weight and accumulate the results. This operation is performed for each input
         * connection, and the results are added together to obtain the final weighted sum.
         *
         * This step is essential because it captures the combined influence of each input
         * on the neuron's output. The weighted sum serves as the input to the activation
         * function (such as the sigmoid function) to determine the neuron's activation
         * level and subsequently its output.
         */
        for ($i = 0; $i < count($inputs); $i++) {
            $sum += $inputs[$i] * $this->weights[$i];
        }

        return $this->sigmoid($sum);
    }

    /**
     * The sigmoid function introduces non-linearity to the neural network.
     * Without non-linear activation functions, a neural network would be
     * limited to representing linear relationships between inputs and outputs.
     * By applying the sigmoid function, the neural network can capture and
     * model complex non-linear relationships, enabling it to learn and generalize
     * from non-linear patterns in the data.
     *
     * Formula: sigmoid(x) = 1 / (1 + exp(-x))
     */
    private function sigmoid($x) {
        return 1 / (1 + exp(-$x));
    }
}