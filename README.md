# PyTorch Seq2Seq

## Примечание. Примеры в этом репозитории работают только с torchtext 0.9 или выше, для которого требуется PyTorch 1.8 или выше. Если вы используете torchtext 0.8, то воспользуйтесь [этой](https://github.com/bentrevett/pytorch-seq2seq/tree/torchtext08) веткой

Этот репозиторий содержит материалы полезные для понимания работы глубоких нейронный сетей sequence-to-sequence (seq2seq) и реализации этих моделей с помощью [PyTorch](https://github.com/pytorch/pytorch) 1.8, [torchtext](https://github.com/pytorch/text) 0.9 и [spaCy](https://spacy.io/) 3.0, под Python 3.8. Материалы расположены в эволюционном порядке: от простой и неточной модели к сложной и обладающей наибольшей точностью. 

**Если вы обнаружите какие-либо ошибки или не согласны с любым из объяснений, пожалуйста, не стесняйтесь обращаться и [сообщать о проблеме/предложении](https://github.com/bentrevett/pytorch-seq2seq/issues/new). Приветствуется как положительная, так и отрицательная критика!**

## Подготовка перед началом работы

Если запускаете в Google Colaboratory, то, возможно, у Вас уже всё есть.

Чтобы установить PyTorch смотрите инструкцию по установке в [PyTorch website](pytorch.org).

Установите torchtext:

``` bash
pip install torchtext
```

Будет использована библиотека spaCy для токенизации данных. Чтобы установить spaCy, следуйте инструкциям [здесь](https://spacy.io/usage/), убедившись, что были установлены английская и немецкая модели:

``` bash
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

## Разделы

Название каждого раздела, за исключением 4-го, соответствует названию породившей его статьи.

* 1 - [Sequence to Sequence Learning with Neural Networks](https://github.com/vasiliyeskin/bentrevett-pytorch-seq2seq_ru/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vasiliyeskin/bentrevett-pytorch-seq2seq_ru/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb)
  
  В этом разделе рассматривается процесс работы над моделью seq2seq с помощью PyTorch и torchtext. Мы начнём с основ сетей seq2seq с использованием модели кодировщик-декодеровщик (кодер-декодер), реализуем эту модель в PyTorch с использованием torchtext для выполнения всей тяжелой работы, связанной с обработкой текста. Сама модель будет основана на реализации [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215), которая использует многослойные LSTM сети.

* 2 - [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://github.com/vasiliyeskin/bentrevett-pytorch-seq2seq_ru/blob/master/2%20-%20Learning%20Phrase%20Representations%20using%20RNN%20Encoder-Decoder%20for%20Statistical%20Machine%20Translation.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vasiliyeskin/bentrevett-pytorch-seq2seq_ru/blob/master/2%20-%20Learning%20Phrase%20Representations%20using%20RNN%20Encoder-Decoder%20for%20Statistical%20Machine%20Translation.ipynb)

  Теперь, когда мы познакомились с базовым рабочим процессом модели seq2seq, в этом разделе сосредоточимся на улучшении полученных результатов. Основываясь на наших знаниях о PyTorch и torchtext, полученных из первой части, мы рассмотрим вторую модель, в которой решена проблема сжатия информации, возникающая в модели кодера-декодера. Эта модель будет основана на реализации  [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078), которая использует GRU сеть.

* 3 - [Neural Machine Translation by Jointly Learning to Align and Translate](https://github.com/vasiliyeskin/bentrevett-pytorch-seq2seq_ru/blob/master/3%20-%20Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vasiliyeskin/bentrevett-pytorch-seq2seq_ru/blob/master/3%20-%20Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate.ipynb)
  
  Далее мы познакомимся с таким понятием как внимание, реализовав модель из статьи [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473). Эта особенность новой модели разрешает проблему сжатия информации, позволяя декодеру «оглядываться» на входное предложение, используя для этого векторы контекста, которые являются взвешенными суммами скрытых состояний кодера. Веса для этих взвешенных сумм вычисляются с помощью механизма внимания. Как итог, декодер учится обращать внимание на наиболее важные слова в конструкции входного предложения при построении выходного.

* 4 - [Уплотнённые Последовательности, Маскировка, Использование модели и BLEU](https://github.com/vasiliyeskin/bentrevett-pytorch-seq2seq_ru/blob/master/4%20-%20Packed%20Padded%20Sequences%2C%20Masking%2C%20Inference%20and%20BLEU.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vasiliyeskin/bentrevett-pytorch-seq2seq_ru/blob/master/4%20-%20Packed%20Padded%20Sequences%2C%20Masking%2C%20Inference%20and%20BLEU.ipynb)

  В этой части мы улучшим предыдущую архитектуру модели, добавив *уплотнённые последовательности* и *маскировку*. Эти два метода обычно используются в обработке естественного языка (NLP). Уплотнённые последовательности позволяют нам обрабатывать только элементы входного предложения нашей рекуррентной сетью. Маскировка используется для того, чтобы заставить модель игнорировать определенные элементы, на которые мы не хотим, чтобы она обращала внимание, например, на дополненные\вспомогательные слова. Вместе они дают некоторый прирост производительности. Кроме того, будет рассмотрен простой способ использования модели для вывода, позволяющий получать перевод любого предложения, которое пропускается через модель. В дополнение к этому, будет реализован способ просмотра значений элементов вектора внимания для этих переводов в исходной последовательности. Наконец, будет показано, как вычислить метрику BLEU по выданным переводам.

* 5 - [Convolutional Sequence to Sequence Learning](https://github.com/vasiliyeskin/bentrevett-pytorch-seq2seq_ru/blob/master/5%20-%20Convolutional%20Sequence%20to%20Sequence%20Learning.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vasiliyeskin/bentrevett-pytorch-seq2seq_ru/blob/master/5%20-%20Convolutional%20Sequence%20to%20Sequence%20Learning.ipynb)

  Наконец, мы отойдём от моделей seq2seq на основе рекуррентных сетей и реализуем эту сеть полностью на основе свёрточной модели. Одним из недостатков рекуррентных сетей — это то, что они являются последовательными. То есть, прежде чем слово будет обработано рекуррентной сетью, все предыдущие слова должны быть пропущены через неё. Свёрточные модели можно полностью распараллелить, что позволяет обучать их намного быстрее. Мы будем реализовывать модель [Convolutional Sequence to Sequence](https://arxiv.org/abs/1705.03122), которая использует несколько свёрточных слоев как в кодере, так и в декодере с включённым механизмом внимания. 

* 6 - [Attention Is All You Need](https://github.com/vasiliyeskin/bentrevett-pytorch-seq2seq_ru/blob/master/6%20-%20Attention%20is%20All%20You%20Need.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vasiliyeskin/bentrevett-pytorch-seq2seq_ru/blob/master/6%20-%20Attention%20is%20All%20You%20Need.ipynb)

  Продолжая реализовывать модели, не базирующиеся на рекуррентных сетях, создадим модель Transformer из [Attention Is All You Need](https://arxiv.org/abs/1706.03762). Эта сеть основана исключительно на механизме внимания с особой реализацией "многонаправленного внимания". Кодер и декодер состоят из нескольких уровней, каждый из которых состоит из подслоев "многонаправленного внимания" и Positionwise Feedforward. Эта модель в настоящее время используется во многих современных задачах последовательного обучения и передачи знаний. 

## Список литературы и дополнительных источников

Ниже некоторые ссылки на работы, которые помогли при создании этих учебных материалов. Некоторые из них могут быть устаревшими.

- https://github.com/spro/practical-pytorch
- https://github.com/keon/seq2seq
- https://github.com/pengshuang/CNN-Seq2Seq
- https://github.com/pytorch/fairseq
- https://github.com/jadore801120/attention-is-all-you-need-pytorch
- http://nlp.seas.harvard.edu/2018/04/03/attention.html
- https://www.analyticsvidhya.com/blog/2019/06/understanding-transformers-nlp-state-of-the-art-models/
- Николенко С.И., Кадурин А., Архангельская Е.В. Глубокое обучение. Погружение в мир нейронных сетей. Санкт-Петербург: Питер. 2020. 481 с.
- Гудфеллоу Я., Бенджио И., Курвилль А. Глубокое обучение. Москва: ДМК-Пресс. 2018. 652 с.
