"use strict";

// Neural Network
// Neural Network
// Neural Network
// Neural Network
// Neural Network
// Neural Network
// Neural Network
// Neural Network

class NeuralNetwork {
    constructor(numImputs, numHidden, numOutputs) {
        this._numImputs = numImputs;
        this._numHidden = numHidden;
        this._numOutputs = numOutputs;

        this._inputs = [];
        this._hidden = [];

        this._bias0 = new Matrix(1, this._numHidden);
        this._bias1 = new Matrix(1, this._numOutputs);

        this._weights0 = new Matrix(this._numImputs, this._numHidden);
        this._weights1 = new Matrix(this._numHidden, this._numOutputs);

        // Randomize Initial weights
        this._bias0.randomWeights();
        this._bias1.randomWeights();
        this._weights0.randomWeights();
        this._weights1.randomWeights();
    }

    get hidden() {
        return this._hidden;
    }

    set hidden(hidden) {
        this._hidden = hidden
    }

    get imputs() {
        return this._inputs;
    }

    set imputs(imputs) {
        this._imputs = imputs
    }

    get bias0() {
        return this._bias0;
    }
    set bias0(bias0) {
        this._bias0 = bias0;
    }

    get bias1() {
        return this._bias1;
    }
    set bias1(bias1) {
        this._bias1 = bias1;
    }

    get weights0() {
        return this._weights0;
    }
    set weights0(weights0) {
        this._weights0 = weights0;
    }

    get weights1() {
        return this._weights1;
    }
    set weights1(weights1) {
        this._weights1 = weights1;
    }

    feedForward(inputArray) {
        this.inputs = Matrix.convertFromArray(inputArray);

        this.hidden = Matrix.dot(this.inputs, this.weights0);
        // Applying Bias to neuron
        this.hidden = Matrix.add(this.hidden , this.bias0);
        // Apply sigmoid function to each cell
        this.hidden = Matrix.map(this.hidden, x => sigmoid(x));
        
        let outputs = Matrix.dot(this.hidden, this.weights1);
        // Applying Bias to neuron
        outputs = Matrix.add(outputs , this.bias1);
        // Apply sigmoid function to each cell
        outputs = Matrix.map(outputs, x => sigmoid(x));

        return outputs;
    }

    // TRain Neurons

    train(inputArray, targetArray) {
        let outputs = this.feedForward(inputArray);
        let targets = Matrix.convertFromArray(targetArray);
        let outputErrors = Matrix.substract(targets, outputs);

        // calculate deltas ( errors * derivative of the output)
        let outputDerivatives = Matrix.map(outputs, x => sigmoid(x, true))
        let outputDeltas = Matrix.multiply(outputErrors, outputDerivatives)

        // Calculate hidden layer errors (delta "dot" transpose of weights1)
        let weights1Tansposed = Matrix.transpose(this.weights1);
        let hiddenErrors = Matrix.dot(outputDeltas, weights1Tansposed);

        // Calculate the hidden deltas ( errors * derivatives of weights1)
        let hiddenDerivatives = Matrix.map(this.hidden, x => sigmoid(x, true))
        let hiddenDeltas = Matrix.multiply(hiddenErrors, hiddenDerivatives);

        // Update the weights ( add transpose of layes "dot" deltas)
        let hiddenTranspose = Matrix.transpose(this.hidden);
        this.weights1 = Matrix.add(this.weights1, Matrix.dot(hiddenTranspose, outputDeltas));
        let inputsTranspose = Matrix.transpose(this.inputs);
        this.weights0 = Matrix.add(this.weights0, Matrix.dot(inputsTranspose, hiddenDeltas));

        // Update Biases
        this.bias1 = Matrix.add(this.bias1 , outputDeltas)
        this.bias0 = Matrix.add(this.bias0 , hiddenDeltas)

    }

}

function sigmoid(x, derivative = false) {
    if (derivative) {
        return x * (1 - x); // where x = sigmoid(x)
    }
    return 1 / (1 + Math.exp(-x));
}

// Matrix Functions
// Matrix Functions
// Matrix Functions
// Matrix Functions
// Matrix Functions
// Matrix Functions
// Matrix Functions
// Matrix Functions
// Matrix Functions

class Matrix {
    constructor(rows, cols, data = []) {
        this._rows = rows;
        this._cols = cols;
        this._data = data;

        // initialise with zeroes if no data provided
        if (data == null || data.length == 0) {
            this._data = [];
            for (let i = 0; i < this._rows; i++) {
                this._data[i] = [];
                for (let j = 0; j < this._cols; j++) {
                    this._data[i][j] = 0;
                }
            }
        } else {
            // check data integrity
            if (data.length != rows || data[0].length != cols) {
                throw new Error("Incorrect data dimensions!");
            }
        }
    }

    get rows() { return this._rows; }

    get cols() { return this._cols; }

    get data() { return this._data; }

    static add(m0, m1) {
        Matrix.checkDimensions(m0, m1);
        let m = new Matrix(m0.rows, m0.cols);
        for (let i = 0; i < m.rows; i++) {
            for (let j = 0; j < m.cols; j++) {
                m.data[i][j] = m0.data[i][j] + m1.data[i][j];
            }
        }
        return m;
    }

    // Not dot product
    static multiply(m0, m1) {
        Matrix.checkDimensions(m0, m1);
        let m = new Matrix(m0.rows, m0.cols);
        for (let i = 0; i < m.rows; i++) {
            for (let j = 0; j < m.cols; j++) {
                m.data[i][j] = m0.data[i][j] * m1.data[i][j];
            }
        }
        return m;
    }

    // dot product of two matrices
    static dot(m0, m1) {

        if (m0.cols != m1.rows) {
            throw new Error("Matrices are not \"dot\" compatible!");
        }
        let m = new Matrix(m0.rows, m1.cols);
        for (let i = 0; i < m.rows; i++) {
            for (let j = 0; j < m.cols; j++) {
                let sum = 0;
                for (let k = 0; k < m0.cols; k++) {
                    sum += m0.data[i][k] * m1.data[k][j];
                }
                m.data[i][j] = sum;
            }
        }
        return m;
    }

    static substract(m0, m1) {
        Matrix.checkDimensions(m0, m1);
        let m = new Matrix(m0.rows, m0.cols);
        for (let i = 0; i < m.rows; i++) {
            for (let j = 0; j < m.cols; j++) {
                m.data[i][j] = m0.data[i][j] - m1.data[i][j];
            }
        }
        return m;
    }

    static convertFromArray(arr) {
        return new Matrix(1, arr.length, [arr]);
    }

    // Apply function to each cell of matrix
    static map(m0, myFunction) {
        let m = new Matrix(m0.rows, m0.cols);
        for (let i = 0; i < m.rows; i++) {
            for (let j = 0; j < m.cols; j++) {
                m.data[i][j] = myFunction(m0.data[i][j])
            }
        }
        return m;
    }

    // find transpose of
    static transpose(m0) {
        let m = new Matrix(m0.cols, m0.rows);
        for (let i = 0; i < m0.rows; i++) {
            for (let j = 0; j < m0.cols; j++) {
                m.data[j][i] = m0.data[i][j]
            }
        }
        return m;
    }

    static checkDimensions(m0, m1) {
        if (m0.rows != m1.rows || m0.cols != m1.cols) {
            throw new Error("Matrices are of different Sizes");
        }
    }

    randomWeights() {
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                this.data[i][j] = Math.random() * 2 - 1;
            }
        }
    }
}




// Neural Nerwork constans
const NUM_IMPUTS = 2;
const NUM_HIDDEN = 5;
const NUM_OUTPUTS = 1;
const NUM_SAMPLES = 100000;

// //////////////////////////////////////////
// //////////////////////////////////////////
// //////////////////////////////////////////

let neuralNetowk = new NeuralNetwork(NUM_IMPUTS, NUM_HIDDEN, NUM_OUTPUTS)

//Train the network
for (let i = 0; i < NUM_SAMPLES; i++) {
    // Test XOR Gate Logic
    //0 0 = 0
    //0 1 = 1
    //1 0 = 1
    //1 1 = 0
    let input0 = Math.round(Math.random())
    let input1 = Math.round(Math.random())
    let output = (input0 == input1) ? 0 : 1;
    neuralNetowk.train([input0, input1], [output])
}
// Test output
console.log("0 0 = " + neuralNetowk.feedForward([0, 0]).data)
console.log("1 0 = " + neuralNetowk.feedForward([1, 0]).data)
console.log("0 1 = " + neuralNetowk.feedForward([0, 1]).data)
console.log("1 1 = " + neuralNetowk.feedForward([1, 1]).data)

// //////////////////////////////////////////
// //////////////////////////////////////////
// //////////////////////////////////////////
// //////////////////////////////////////////
// //////////////////////////////////////////
