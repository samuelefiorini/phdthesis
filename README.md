# Challenges in biomedical data science:
## *data-driven solutions to clinical questions*

PhD Thesis, Samuele Fiorini, Februrary 2018. Defended on April 25, 2018 at University of Genoa - [DIBRIS](https://www.dibris.unige.it/en/).

*Content:*

- `tex` has everything you need to compile my PhD Thesis using `pdflatex` (figures are automatically fetched from `images`)
- `src` has all the code needed to reproduce the figures of the Thesis using `python` scripts or `jupyter notebook`
- `presentation` has the Keynote presentation I used to defend my thesis


*Contacts*:

`samuele [dot] fiorini [at] dibris.unige.it`

*BibTex entry*:

```
@phdthesis{fiorini2018challenges,
  title={Challenges in biomedical data science: data-driven solutions to clinical questions},
  author={Fiorini, Samuele},
  howpublished="\url{http://hdl.handle.net/11567/930182}"
  year={2018},
  school={University of Genoa},
}
```

____
### Abstract
Data are influencing every aspect of our lives, from our work activities, to our spare time and even to our health.
In this regard, medical diagnosis and treatments are often supported by quantitative measures and observations, such as laboratory tests, medical imaging or genetic analysis.
In medicine, as well as in several other scientific domains, the amount of data involved in each decision-making process has become overwhelming.
The complexity of the phenomena under investigation and the scale of modern data collections has long superseded human analysis and insights potential.
Therefore, a new scientific branch that simultaneously addresses statistical and computational challenges is rapidly emerging, and it is known as data science.

Data science is the evolving cross-disciplinary field that, borrowing concepts from several scientific areas, aims at devising data-driven decision making strategies for real-world problems.
Data science differs from classical applied statistics as it generally combines mathematical background with advanced computer science and thorough domain knowledge.
Following the data science philosophy, it is possible to ask the right questions to the data and to find statistically sound answers in a reasonable amount of time.

Machine learning can be seen as one of the main components of data science. The aim of machine learning is to devise algorithms that can recognize and exploit hidden patterns in some set of data in order to formulate an accurate prediction strategy that holds for current and future data as well. Thanks to machine learning it is now possible to achieve automatic solutions to complex tasks with little human supervision.
As of today, machine learning is the workhorse of data science.

This thesis revolves around the application of machine learning and data science concepts to solve biomedical and clinical challenges. In particular, after a preliminary overview of the main data science and machine learning concepts and techniques, we will see the importance of exploratory data analysis and how it can be easily performed on any structured input dataset. Moreover, we will see that, with sparsity-enforcing linear models, it is possible to predict the age of an individual from a set of molecular biomarkers measured from peripheral blood. Furthermore, we will present a nonlinear temporal model that accurately predicts the disease evolution in multiple sclerosis patients from a set of inexpensive and patient-friendly measures repeated in time. Finally, we will see how with a continuous glucose monitor and a kernel machine it is possible to accurately forecast the glycemic level of type 1 and type 2 diabetic patients, in order to improve their treatment.

With the final aim of devising actionable solutions, that are ready to be applied in clinical contexts, the predictive performance of the data-driven models proposed throughout the chapters of this thesis is rigorously assessed exploiting bias-aware cross-validation schemes.
