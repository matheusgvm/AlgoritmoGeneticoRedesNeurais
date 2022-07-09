import random
import matplotlib.pyplot as plt
import RedeNeural


class Individuo:
    def __init__(self, num_camadas_ocultas, num_neuronios_por_camada_oculta, batch_size, taxa_mutacao):
        self.taxa_mutacao = taxa_mutacao
        self.num_camadas_ocultas = num_camadas_ocultas
        self.num_neuronios_por_camada_oculta = num_neuronios_por_camada_oculta
        self.num_epocas = None
        self.batch_size = batch_size
        self.score_funcao_objetivo = None
        self.ranking_linear = None

    def mutacao(self):
        if random.uniform(0, 1) < self.taxa_mutacao:
            fator = random.choice([-1, 1])
            self.num_camadas_ocultas += fator
            if self.num_camadas_ocultas < 0:
                self.num_camadas = 0
            elif fator > 0:
                self.num_neuronios_por_camada_oculta.append(random.randint(1, 16))

            else:
                self.num_neuronios_por_camada_oculta.pop()

        for i in range(self.num_camadas_ocultas):
            if random.uniform(0, 1) < self.taxa_mutacao:
                self.num_neuronios_por_camada_oculta[i] += random.choice([-1, 1])
                if self.num_neuronios_por_camada_oculta[i] < 1:
                    self.num_neuronios_por_camada_oculta[i] = 1

        if random.uniform(0, 1) < self.taxa_mutacao:
            self.batch_size = random.choice([4, 8, 16, 32, 64, 128, 256, 512, 1024])


# De 1 a 100
def ranking_linear(x):
    return 1 + (99 - 1) * ((100 - x) / (100 - 1))


class Populacao:
    def __init__(self, taxa_mutacao, tamanho_populacao, criar_rede_neural):
        self.tamanho_populacao = tamanho_populacao
        self.taxa_mutacao = taxa_mutacao
        self.individuos = []
        self.criar_rede_neural = criar_rede_neural

    def inicializar_populacao_aleatoriamente(self):
        self.individuos = []
        for i in range(self.tamanho_populacao):
            num_camadas_ocultas = random.randint(1, 16)
            num_neuronios_por_camada_oculta = []
            for j in range(num_camadas_ocultas):
                num_neuronios_por_camada_oculta.append(random.randint(1, 32))
            self.individuos.append(Individuo(num_camadas_ocultas, num_neuronios_por_camada_oculta,
                                             random.choice([4, 8, 16, 32, 64, 128, 256, 512, 1024]),
                                             self.taxa_mutacao))

    def set_score_funcao_objetivo(self):
        for individuo in self.individuos:
            lista_retorno = self.criar_rede_neural.executar(individuo.num_camadas_ocultas,
                                                                                individuo.num_neuronios_por_camada_oculta,
                                                                                individuo.batch_size)
            individuo.score_funcao_objetivo = - lista_retorno[0]
            individuo.num_epocas = lista_retorno[1]

    def set_ranking_linear(self):
        self.individuos.sort(key=lambda x: x.score_funcao_objetivo, reverse=True)
        for i, individuo in enumerate(self.individuos):
            individuo.ranking_linear = ranking_linear(i)

    def selecionar_individuo_roleta(self, max_roleta):
        n_aleatorio = random.uniform(0, max_roleta)
        soma_roleta = 0
        for individuo in self.individuos:
            soma_roleta += individuo.ranking_linear
            if soma_roleta >= n_aleatorio:
                return individuo

    def get_melhor_individuo(self):
        self.individuos.sort(key=lambda x: x.score_funcao_objetivo, reverse=True)
        return self.individuos[0]

    def get_pior_individuo(self):
        self.individuos.sort(key=lambda x: x.score_funcao_objetivo, reverse=True)
        return self.individuos[-1]

    def get_individuo_mediana(self):
        self.individuos.sort(key=lambda x: x.score_funcao_objetivo, reverse=True)
        return self.individuos[int(self.tamanho_populacao / 2)]

    def get_media_individuo(self):
        soma = sum(individuo.score_funcao_objetivo for individuo in self.individuos)
        return soma / self.tamanho_populacao

    def gerar_nova_populacao(self):
        nova_populacao = Populacao(self.taxa_mutacao, self.tamanho_populacao, self.criar_rede_neural)
        # Selecionar individuos
        max_roleta = sum(individuo.ranking_linear for individuo in self.individuos)
        for i in range(int(self.tamanho_populacao / 2)):
            individuo_1 = self.selecionar_individuo_roleta(max_roleta)
            individuo_2 = self.selecionar_individuo_roleta(max_roleta)

            # Faz o crossover
            filho_1 = self.crossover(individuo_1, individuo_2)
            filho_2 = self.crossover(individuo_1, individuo_2)

            # Faz a mutacao
            filho_1.mutacao()
            filho_2.mutacao()

            # Adiciona os filhos na nova populacao
            nova_populacao.individuos.append(filho_1)
            nova_populacao.individuos.append(filho_2)

        return nova_populacao

    def crossover(self, individuo1, individuo2):
        a = random.uniform(0, 1)
        num_camadas_ocultas = round(((individuo1.num_camadas_ocultas * a) + (individuo2.num_camadas_ocultas * (1-a))) / 2)
        num_neuronios_por_camada_oculta = []

        for i in range(num_camadas_ocultas):
            if individuo1.num_camadas_ocultas < i + 1:
                num_neuronios_por_camada_oculta.append(individuo2.num_neuronios_por_camada_oculta[i])
            elif individuo2.num_camadas_ocultas < i + 1:
                num_neuronios_por_camada_oculta.append(individuo1.num_neuronios_por_camada_oculta[i])
            else:
                a = random.uniform(0, 1)
                num_neuronios_por_camada_oculta.append(round(((individuo1.num_neuronios_por_camada_oculta[i] * a) + (individuo2.num_neuronios_por_camada_oculta[i] * (1-a))) / 2))

        a = random.uniform(0, 1)
        batch_size = ((individuo1.batch_size * a) + (individuo2.batch_size * (1 - a))) / 2
        batch_size = min([4, 8, 16, 32, 64, 128, 256, 512, 1024], key=lambda x:abs(x-batch_size))

        return Individuo(num_camadas_ocultas, num_neuronios_por_camada_oculta, batch_size, self.taxa_mutacao)


class Experimento:
    def __init__(self, n_geracoes, taxa_mutacao, tamanho_populacao):
        self.n_geracoes = n_geracoes
        self.geracoes = []
        self.taxa_mutacao = taxa_mutacao
        self.tamanho_populacao = tamanho_populacao
        self.criar_rede_neural = None

    def executar_experimento(self):

        # Criar objeto
        criar_rede_neural = RedeNeural.CriarRedeNeural('winequality-red.csv')

        populacao_inicial = Populacao(self.taxa_mutacao, self.tamanho_populacao, criar_rede_neural)
        self.criar_rede_neural = criar_rede_neural
        populacao_inicial.inicializar_populacao_aleatoriamente()
        populacao_inicial.set_score_funcao_objetivo()
        populacao_inicial.set_ranking_linear()
        self.geracoes.append(populacao_inicial)

        populacao_atual = populacao_inicial

        for i in range(self.n_geracoes):
            populacao_atual = populacao_atual.gerar_nova_populacao()
            populacao_atual.set_score_funcao_objetivo()
            populacao_atual.set_ranking_linear()
            self.geracoes.append(populacao_atual)
            print("Geracao: " + str(i + 1))

    def exibir_melhor_individuo(self):
        melhor_individuo = self.geracoes[-1].get_melhor_individuo()
        print()
        print("Melhor indivíiduo")
        print("Perda: " + str(melhor_individuo.score_funcao_objetivo))
        print("Número de camadas ocultas: " + str(melhor_individuo.num_camadas_ocultas))
        for i in range(melhor_individuo.num_camadas_ocultas):
            print("Camada " + str(i+1) + ": " + str(melhor_individuo.num_neuronios_por_camada_oculta[i]) + " neurônio(s)")
        print("Última camada: 1 neurônio")
        print("Número de épocas: " + str(melhor_individuo.num_epocas))
        print("Tamanho do batch: " + str(melhor_individuo.batch_size))


    def exibir_curva_desempenho(self):
        plt.plot([i for i in range(self.n_geracoes)],
                 [self.geracoes[i].get_melhor_individuo().score_funcao_objetivo for i in range(self.n_geracoes)],
                 label="Melhor individuo", color="green")
        plt.plot([i for i in range(self.n_geracoes)],
                 [self.geracoes[i].get_media_individuo() for i in range(self.n_geracoes)], label="Média",
                 color="yellow")
        plt.plot([i for i in range(self.n_geracoes)],
                 [self.geracoes[i].get_pior_individuo().score_funcao_objetivo for i in range(self.n_geracoes)],
                 label="Pior individuo", color="red")
        plt.xlabel('Geração')
        plt.ylabel('Score')
        plt.title('Curva de desempenho')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    experimento = Experimento(n_geracoes=3, taxa_mutacao=0.01, tamanho_populacao=100)
    experimento.executar_experimento()
    experimento.exibir_melhor_individuo()
    experimento.exibir_curva_desempenho()
    print("Número de modelos criados: " + str(experimento.criar_rede_neural.num_execucoes))
