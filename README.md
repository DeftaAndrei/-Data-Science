Ex2 Eu nu m-am aptinut si am incercat sa fac 2 probleme ca daca tot am stat sa citesc atat de mult despre apache parquet si am gasit tot ce trebuia pe https://parquet.apache.org/ si pot sa zic ca m-am inspirat si dupa https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html#sphx-glr-auto-examples-inspection-plot-linear-model-coefficient-interpretation-py
   Pot  sa zic ca red ca mi-ar placea sa lucrez cu baze de datre intradevar cred ca pot sa ma situez la mid pentru inceput trebuie sa stau sa invat foarte multe dar cred ca voi puteti sa ma ghidati foarte bine in aceasta aventura
Acesta este biroul meu . Am devenit mult mai productiv decand am investit in ce iubesc sa fac 
![WhatsApp Image 2025-03-16 at 20 31 58_27d8945b](https://github.com/user-attachments/assets/834a5196-6f90-42e2-a1fe-0d0aea90b4c9)

# Potrivirea și gruparea site-urilor web în funcție de similitudinea logourilor

Acest proiect implementează o soluție pentru potrivirea și gruparea site-urilor web în funcție de similitudinea logourilor lor. Soluția utilizează mai multe abordări pentru a extrage caracteristicile logourilor și a le grupa, inclusiv tehnici tradiționale de procesare a imaginilor și tehnici de învățare profundă.

## Descrierea soluției

Soluția implementează trei abordări diferite pentru potrivirea logourilor:

1. **Abordarea tradițională** (`logo_matcher.py`): Utilizează tehnici clasice de procesare a imaginilor, cum ar fi histograme de culori și caracteristici HOG (Histogram of Oriented Gradients), pentru a extrage caracteristicile logourilor.

2. **Abordarea de învățare profundă** (`deep_logo_matcher.py`): Utilizează un model pre-antrenat ResNet-50 pentru a extrage caracteristicile profunde ale logourilor.

3. **Abordarea ensemble** (`ensemble_logo_matcher.py`): Combină caracteristicile extrase de abordările tradiționale și de învățare profundă pentru a obține rezultate mai bune.

Pentru gruparea logourilor similare, toate abordările utilizează algoritmul DBSCAN (Density-Based Spatial Clustering of Applications with Noise), care este potrivit pentru această sarcină deoarece nu necesită specificarea numărului de clustere în avans și poate identifica outlier-i.

## Structura proiectului

- `logo_matcher.py`: Implementarea abordării tradiționale
- `deep_logo_matcher.py`: Implementarea abordării de învățare profundă
- `ensemble_logo_matcher.py`: Implementarea abordării ensemble
- `inspect_parquet.py`: Script pentru inspectarea structurii fișierului parquet
- `run_all.py`: Script principal care rulează toate abordările și compară rezultatele
- `results/`: Director pentru rezultate (creat automat)

## Cerințe

Pentru a rula acest proiect, aveți nevoie de următoarele biblioteci Python:

```
pandas
numpy
pillow
scikit-learn
matplotlib
opencv-python
torch
torchvision
pyarrow
```

Puteți instala toate dependențele folosind:

```bash
pip install pandas numpy pillow scikit-learn matplotlib opencv-python torch torchvision pyarrow
```

## Cum să rulați soluția

1. Asigurați-vă că aveți fișierul `logos.snappy(1).parquet` în directorul curent.

2. Pentru a inspecta structura fișierului parquet:

```bash
python inspect_parquet.py
```

3. Pentru a rula abordarea tradițională:

```bash
python logo_matcher.py
```

4. Pentru a rula abordarea de învățare profundă:

```bash
python deep_logo_matcher.py
```

5. Pentru a rula abordarea ensemble:

```bash
python ensemble_logo_matcher.py
```

6. Pentru a rula toate abordările și a compara rezultatele:

```bash
python run_all.py
```

## Rezultate

Rezultatele vor fi afișate în consolă și salvate în directorul `results/`. Acestea includ:

- Grupurile de site-uri web cu logourile similare
- Vizualizări ale grupurilor pentru fiecare abordare
- Comparații între diferitele abordări

## Abordarea tehnică

### Extragerea caracteristicilor

#### Abordarea tradițională
- Histograme de culori (RGB)
- Caracteristici HOG simplificate

#### Abordarea de învățare profundă
- Caracteristici extrase de modelul ResNet-50 pre-antrenat

#### Abordarea ensemble
- Combinarea caracteristicilor tradiționale și profunde

### Gruparea

Toate abordările utilizează algoritmul DBSCAN pentru grupare, cu următorii parametri:
- `eps`: Distanța maximă între două eșantioane pentru a fi considerate în același cluster
- `min_samples`: Numărul minim de eșantioane într-un cluster

Abordarea ensemble evaluează automat diferite valori pentru acești parametri și selectează cea mai bună configurație.

### Evaluarea

Rezultatele sunt evaluate folosind:
- Scorul Rand ajustat pentru a compara grupurile obținute de diferitele abordări
- Numărul de grupuri și outlier-i pentru fiecare abordare
- Matricea de confuzie pentru a analiza suprapunerea între grupuri

## Scalabilitate

Soluția este proiectată pentru a fi scalabilă:
- Utilizează procesare paralelă pentru extragerea caracteristicilor
- Poate fi adaptată pentru a procesa seturi de date mai mari
- Poate fi implementată într-un mediu distribuit pentru procesarea la scară largă

## Îmbunătățiri posibile

- Utilizarea altor modele de învățare profundă pentru extragerea caracteristicilor
- Implementarea altor algoritmi de grupare
- Optimizarea parametrilor folosind tehnici de căutare în grilă sau optimizare bayesiană
- Implementarea unei interfețe web pentru vizualizarea rezultatelor
- Utilizarea tehnicilor de augmentare a datelor pentru a îmbunătăți robustețea


  O sa va pun o poza pentru cel de-al doilea proiect
  ![image](https://github.com/user-attachments/assets/8cbc9d90-8877-4079-a2a1-8d70626bea08)
  


## Autor Defta Andrei Robert
Va multumesc daca ati ajuns pana aici 

Acest proiect a fost dezvoltat ca parte a procesului de recrutare pentru Veridion. 
