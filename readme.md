O arquivo perceptron.py é efetivamente a implementação de uma rede neural perceptron de 1 layer e 1 neurônio.
Sendo capaz de separar duas classes entre si através da análise de dois atributos.

A implementação foi realizada com o uso de orientação a objetos, então inicialmente deve-se criar um objeto da classe Perceptron.
O construtor espera por dois paramatros, o numero máximo de épocas de treinamento e a taxa de aprendizagem.

p = Perceptron(max_training_epoch=100, learning_rate=0.001)

Feito isto, pode-se então treinar a rede através do método train.

p.train(train_values, train_labels)

Durante o treinamento, a cada época será plotado os pontos utilizados para treinamento bem como a reta que representa a fronteira de decisão. 
Para que uma proxima época seja executada, deve-se clicar para fechar o plot, e então um próximo abrirá, representando uma nova iteração.
No método train, utiliza-se o método check_convergence para verificar se houve convergência antes de atingir o numero máximo de epocas configurado.

Quando a convergência for obtida, será plotado o data set de teste e a fronteira de decisão final, e o titulo do plot demonstrará quantas épocas foram necessárias.
Com o treino realizado, pode-se então chamar o método predict que retornará uma lista com as previsões feitas para os valores passados.

predictions = p.predict(test_values)

Por fim, pode-se chamar o métudo accuracy para verificar a acurácia das previsões em relação às labels reais.
Este método espera as labels verdadeiras e as labels previstas e então retorna a acurácia.

p.accuracy(test_labels, predictions)

O formato esperado para values e labels, tanto no treino quanto na previsão é:

[[atributo1 atributo2]
 [atributo1 atributo2]
 [atributo1 atributo2]
 .
 .
 .
]

[label label label ... ]


Ao executar o arquivo perceptron.py (python /path/to/perceptron.py) será executado um treino e teste com dados criados pela função make_blobs da biblioteca scikit-learn.

Ao executar o arquivo iris.py (python /path/to/iris.py) utiliza-se então o iris data set para treino e teste.
Como esse data set tem 4 atributos e 3 classes, utilizou-se apenas 2 atributos (sepal lenth e petal length) e apenas 2 classes (setosa e versicolor)
Desta forma, apenas as 100 primeiras linhas do data set foram utilizadas, onde então foram divididas nos grupos de treino e teste, sendo que o grupo de teste representa 20% do data set enquanto o restante fica no grupo de treino.