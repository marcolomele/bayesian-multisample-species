
with the convention (1  ) 1 ⌘ 1. The closed form expression displayed in (11) is a fundamental
tool to derive the full conditional distributions of the Gibbs sampler described in Section 5. As for
the actual determination of⇧ (n 1 +n 2 )
k , a proof can be found in [3].
5 Algorithm for predictions
Once the probability distribution of the underlying partially exchangeable random partition has
been determined through (10), one can address the issue of predicting m future outcomes of a
certain experiments as mentioned in Section 2. To be more specific, conditional on observed data
X (n i )
i , interest lies in predicting specific features of additional and unobserved samples X (m i |n i )
i ,
for i = 1, 2. Had one solely been interested in estimating L i,j
s,m , in view of Theorem 1 it would
have been enough to determine the single-step prediction (m = 1) and obtain the estimate for a
general m by linearity. However, since we also aim at identifying highest posterior density (HPD)
regions of L i,j
s,m , a general m-step prediction algorithm is mandatory (of which m = 1 represents
a simple special case). Furthermore, the simulation of realizations of X (m i |n i )
i , for i = 1, 2, is of
interest if one is also willing to infer other quantities of interest in species sampling problems such
as, e.g., the number of new species that will be observed or the so-called discovery probability.
See [19]. Finally, note that despite the presentation concerns the multiple-samples case, obvious
modifications allow one to devise an algorithm for the exchangeable case (d = 1).
Our goal is to generate samples X 1,n 1 +1 , . . . , X 1,n 1 +m 1 and X 2,n 2 +1 , . . . , X 2,n 2 +m 2 , conditional
on X (n 1 ) and X (n 2 ) , for any two positive integers m 1 and m 2 . In order to employ (11) one needs
to introduce n 1 + m1 + n 2 + m2 latent variables T 1,1 , . . . , T 1,n 1 +m 1 , T 2,1 , . . . , T 2,n 2 +m 2 , which are
the labels identifying the tables at which the di↵erent costumers are seated in the restaurants. If
the additional m = m1 + m2 data induce j new distinct observations not included in X (n 1 ) and
X (n 2 ) , the determination of the full conditionals follows immediately from the pEPPF, which is
easily deduced from (11). One finds
⇧ (n 1 +n 2 +m)
k (n 1 , n 2 ; `, q)
= 
(` •• )
k+j,0 (` •1 , . . . , `•k+j )
2Y
i=1
 (n i +m i )
` i• ,i (q i,1 , . . . , q i,k+j )
=
Q k+j1
r=1 (✓ 0 + r 0 )
(✓ 0 + 1) ` •• 1
k+jY
t=1
(1   0 ) ` •t 1
Q ` 1• 1
r=1 (✓ + r)
(✓ + 1) n 1 +m 1 1
k+jY
v=1
` 1,v
Y
t=1
(1  ) q 1,v,t 1
⇥
Q ` 2• 1
r=1 (✓ + r)
(✓ + 1) n 2 +m 2 1
k+jY
v=1
` 2,v
Y
t=1
(1  ) q 2,v,t 1 .
(12)
Based on (12) one can devise a Gibbs sampler that generates (T i,1 , . . . , T i,n i ), for i = 1, 2, and
(X i,n i +r , Ti,n i +r ), for r = 1, . . . , m i and i = 1, 2, from their respective full conditionals. Details
are provided for i = 1, the case i = 2 being identical with the appropriate adaptations. If V is a
variable that is a function of (T i,1 , . . . , T i,n i +m i ) and of (X i,n i +1 , . . . , X i,n i +m i ), use V r to denote
the generic value of the variable V after removal of T i,r , for r = 1, . . . , n i , and of (X i,r , Ti,r ), for
r = n i + 1, . . . , n i + mi .
(1) At t = 0, start from an initial configuration X (0)
i,n i +1 , . . . , X (0)
i,n i +m i and T (0)
i,1 , . . . , T (0)
i,n i +m i , for
i = 1, 2.
8
(2) At iteration t  1,
(2.a) With X 1,r = X ⇤
h generate latent variables T (t)
1,r , for r = 1, . . . , n i , from
P(T 1,r = “new”| · · · ) / w h,r
(✓ + ` r
1• )
(` r
•• + ✓ 0 ) ,
P(T 1,r = T ⇤,r
1,h, | · · · ) / (q r
1,h,  ) for  = 1, . . . , `r
1,h ,
where w h,r = ` r
•h   0 if ` r
•h > 0 and w h,r = 1 otherwise. Moreover, T ⇤,r
1,h,1 , . . . , T ⇤,r
1,h,`r
1,h
are
the tables at the first restaurant where the hth dish is served, after the removal of T 1,r .
(2.b) For r = 1, . . . , m 1 , generate (X (t)
n i +r , T (t)
n i +r ) from the following predictive distributions
P(X 1,r = “new”, T 1,r = “new”| · · · ) = (✓ 0 + (k + j r ) 0 )
(✓ + n 1 + m1  1)
(✓ + ` r
1• )
(✓ 0 + ` r
•• )
while, for any h = 1, . . . , k + j r and  = 1, . . . , `r
1,h ,
P(X 1,r = X ⇤,r
h , T 1,r = “new”| · · · ) = (` r
•h   0 )
(✓ + n 1 + m1  1)
(✓ + ` r
1• )
(✓ 0 + ` r
•• ) ,
P(X 1,r = X ⇤,r
1,h , T 1,r = T ⇤,r
1,h, | · · · ) = q r
1,h,  
✓ + n 1 + m1  1 .
6 Illustrations
We are now ready to apply the algorithm devised in Section 5 to face prediction in species sampling
problems. To this end, we assume that the observations originate from d di↵erent populations of
individuals that can be grouped into classes identified by di↵erent types or species. One can think,
for example, of data related to communities of plants or animals from di↵erent species in unknown
proportions. In this case,  ̃p i in (2) is the distribution of the species in the ith community and
similarities between communities motivate dependence across the  ̃p i ’s.
Here we focus on the analysis of genomic data known as Expressed Sequence Tags (ESTs).
These are generated by partially sequencing randomly isolated gene transcripts that have been
converted into cDNA. In very simplified terms, ESTs are tool for gene identification and an EST
sample of size n consists of K n distinct genes, with expression levels, i.e., frequencies, N 1 , . . . , N K n ,
where N 1 + · · · + N K n = n. A large amount of literature, both frequentist and Bayesian, has been
developed for addressing prediction problems related to exchangeable data in several application
areas, most notably in Ecology, Biology and Genomics. ESTs represent an important instance of
genomics application.
Given a basic observed sample of size n and a potential additional sample of size m various types
of prediction problems can be addressed. For instance, one may consider estimation of the number
of new genes arising in the additional sample of size m or the m-discovery probability, which is
the probability of discovering a new gene at the (n + m + 1)th draw, without having observed the
additional sample of size m. The frequentist approach dates back to the pioneering contributions
of Good and Toulmin [12, 13] and has seen countless contributions since then. Among them we
9
mention [2, 5, 24, 30] and references therein. A Bayesian nonparametric approach to this type of
prediction problems in the exchangeable setting was first proposed in [19] and developments to
date are accounted for in [7]. A method for comparing and testing across di↵erent EST libraries
is set forth in [20]. However, species’ prediction problems within a rigorous partially exchangeable
framework have not been considered in the literature yet.
Here we drop the exchangeability assumption and address prediction in the more general and
realistic framework of multiple populations. We consider two di↵erent cDNA libraries of fruits
of a citrus clementina, namely FlavFr1 and RindPdig24, which, for simplicity, we refer to as
FRUIT 1 and FRUIT 2, respectively. The EST sample corresponding to FRUIT 1, X (n 1 )
1 , contains
n 1 = 1593 ESTs with K n 1 = 806 distinct genes, whereas the sample corresponding to FRUIT 2,
X (n 2 )
2 , is made of n 2 = 900 ESTs with K n 2 = 687 unique genes. Moreover, the two libraries
share 183 distinct genes and, in particular, 520 and 317 ESTs of, respectively, FRUIT 1 and
FRUIT 2 refer to these common genes. The details of the two EST samples and the sample
obtained by merging the two are given in Table 1. These data are freely available at the website
http://www.ncbi.nlm.nih.gov/unigene/.
Expression level FRUIT 1 FRUIT 2 FRUITS
1 561 549 905
2 148 99 231
3 37 20 79
4 18 12 32
5 6 4 11
6 5 9
7 12 1 11
8 1 1 4
9 1 6
10 3 1 2
11 1 3
12 2 3
13 1
14 3 1
15 2 1
16 1 2
17 2
19 1
20 1
22 1
23 1 1
24 2
26 1
58 1 1
117 1 1
n 1593 900 2493
Kn 806 687 1310
Table 1: Citrus clementina: EST clustering profile of cDNA libraries of di↵erent fruits. FRUITS
is FRUIT 1 + FRUIT 2.
Given EST data, the main inferential goal consists in prediction of the outcomes of additional
sequencing, in our case from the two clementina libraries. More precisely, we focus on the number
of genes coinciding with new values to be detected in an additional sample of size m, which
can be distinguished into: (a) L 0,0
s,m for s = 1, 2; (b) L 0,1
1,m and L 1,0
2,m ; (c) L 0
1,m = L 0,0
1,m + L 0,1
1,m and
10
L 0
2,m = L 0,0
2,m +L 1,0
2,m . Recall that (a) and (b) were defined right before Theorem 1. Since closed form
expressions for estimators of these quantities are not available under hierarchical nonparametric
models (2), we approximate all the predictions by resorting to the algorithm described in Section 5.
Indeed, at every iteration t, we generate the trajectory X (t)
i,n i +1 , . . . , X (t)
i,n i +m in order to evaluate
the quantities of interest. For example, we have
ˆL 0
s,m = 1
T
TX
t=1
mX
r=1
1 {X s,1 ,...,X s,ns } c (X (t)
s,n s +r )
for s = 1, 2 and m 2 N. Here we compare the predictions obtained in the simple exchangeable case,
in which the quantities are estimated separately for the two datasets, with the results obtained
in the partially exchangeable case. Note that the latter is a more natural choice, since the two
libraries share a high number of genes, and the assumption of partial exchangeability triggers the
borrowing of strength phenomenon across the libraries. The following numerical outputs are based
on 10,000 iterations of the Gibbs sampler after 5,000 burn-in sweeps.
6.1 Independent exchangeable datasets
We first analyze the two libraries separately, which is equivalent to assuming independence of
the  ̃p i ’s and the prediction rule takes on the form displayed in (3). The corresponding model
specification is
(X 1,i , X 2,j ) | ( ̃p 1 ,  ̃p 2 ) iid
⇠  ̃p 1 ⇥  ̃p 2 ,
( ̃p 1 ,  ̃p 2 ) | ( ̃p 1,0 ,  ̃p 2,0 ) ⇠ PY( 1 , ✓ 1 ;  ̃p 1,0 ) ⇥ PY( 2 , ✓ 2 ;  ̃p 2,0 ),
( ̃p 1,0 ,  ̃p 2,0 ) ⇠ PY( 1,0 , ✓ 1,0 ; P 0 ) ⇥ PY( 2,0 , ✓ 2,0 ; P 0 )
and one can rely on a suitable adaptation of the algorithm devised in Section 5 in order to obtain
approximation predictions. We also set independent non-informative priors for ( i,0 , ✓ i,0 ) and
( i , ✓ i ), for i = 1, 2 given by
( i,0 ,  i , ✓ i,0 , ✓ i ) iid
⇠ U (0, 1) ⇥ U (0, 1) ⇥ G(300, 51 ) ⇥ G(300, 5 1 ),
where U (0, 1) stands for the uniform distribution on the interval (0, 1) and G(a, b) denotes the
Gamma distribution with parameters (a, b); the values of ( i,0 ,  i , ✓ i,0 , ✓ i ) are generated through
a Metropolis–Hastings step. In other terms, the two samples are independent and inferences with
data from one sample do not impact inferences concerning the other sample. Given that there are
183 shared observations, the independence assumption is quite restrictive but it serves as a useful
exercise for drawing comparisons with the more appropriate partial exchangeability assumption.
The estimators of L 0
s,m for s = 1, 2 are summarized, for di↵erent sizes of the additional sample
m, in Table 2. In accordance with Theorem 1, ˆL 1,m increases linearly in m, with a slope which
is larger for the second dataset FRUIT 2, consequence of a higher probability of sampling a new
value at step n + 1 for library 2. Finally, the posterior estimates for the parameters of the marginal
PY processes are equal to
(ˆ✓ 1,0 , ˆ 1,0 , ˆ✓ 1 , ˆ 1 ) = (1213.4, 0.4676, 1387.5, 0.0545),
(ˆ✓ 2,0 , ˆ 2,0 , ˆ✓ 2 , ˆ 2 ) = (1428.1, 0.2726, 1543.3, 0.6532). (13)
11
Citrus clementina: FRUIT 1 Citrus clementina: FRUIT 2
m ˆL 0
1,m HPD (95%) ˆL 0
2,m HPD (95%)
200 68.21 (54, 83) 122 (106, 138)
400 136.21 (114, 159) 244 (219, 269)
600 204.28 (175, 235) 366 (331, 401)
800 272.34 (236, 310) 488 (444, 531)
1000 340.37 (297, 385) 610 (557, 662)
1200 408.48 (358, 461) 731 (670, 792)
1400 476.51 (419, 536) 853 (783, 924)
1600 544.71 (481, 611) 975 (897, 1054)
1800 612.74 (542, 687) 1097 (1011, 1185)
2000 680.83 (604, 760) 1219 (1124, 1314)
Table 2: Posterior expected number of new ESTs with corresponding 95% highest posterior density
intervals for FRUIT 1 and FRUIT 2 in the independent exchangeable setting for the HPYP.
Citrus clementina: FRUIT 1 Citrus clementina: FRUIT 2
m ˆL 0
1,m HPD (95%) ˆL 0
2,m HPD (95%)
200 53.21 (41, 66) 85.32 (71, 100)
400 106.45 (87, 126) 170.70 (149, 192)
600 159.69 (136, 184) 256.04 (228, 284)
800 212.99 (185, 242) 341.37 (307, 376)
1000 266.13 (233, 300) 426.74 (387, 467)
1200 319.28 (282, 357) 512.13 (467, 558)
1400 372.45 (331, 414) 597.53 (547, 648)
1600 425.78 (380, 473) 682.86 (627, 739)
1800 479.01 (429, 530) 768.17 (708, 829)
2000 532.37 (479, 587) 853.59 (788, 920)
Table 3: Posterior expected number of new ESTs with corresponding 95% highest posterior density
intervals for FRUIT 1 and FRUIT 2 in the independent exchangeable setting for the HDP.
It is useful to briefly compare the results of HPYP with the more familiar HDP, which arises
by setting  i =  i,0 = 0 for i = 1, 2. The estimated values of L 0
s,m for s = 1, 2 for the HDP
are reported in Table 3. Not surprisingly, given the findings in [19], the Dirichlet process leads to
strong underestimation. Clearly, if the HDP were the appropriate model to use in this case, the
estimates of the  parameters for the two samples would have all been close to 0, whereas it is
clear from (13) that they are not.
6.2 Partially exchangeable samples
The presence of 183 shared genes across the two libraries indicates that a more elaborate model
accounting for dependence among the two samples is appropriate. The hierarchical structure (2),
with d = 2 and  ̃p 1 ,  ̃p 2 and  ̃p 0 as in (9), seems ideally suited to account for interactions among the
12
two samples. This corresponds to assuming the data to be exchangeable within each library and
conditionally independent across the two libraries. Hence, the number of new genes to be detected
in the additional sample for each library depends also on the sample of the other library. Moreover,
there is a single shared set of parameter values (✓ 0 ,  0 , ✓ , ) for which we set independent priors as
follows
( 0 ,  , ✓ , ✓0 ) ⇠ U(0, 1) ⇥ U (0, 1) ⇥ G(300, 51 ) ⇥ G(300, 51 ).
In a similar fashion as in the exchangeable framework, these parameters are generated through
a Metropolis–Hastings step embedded within the Gibbs sampler. The generalized Blackwell–
MacQueen urn scheme in Section 5, then, yields the simulated trajectories that are used to ap-
proximate posterior inferences. The relevant numerical summaries arising from the algorithm are
reproduced in Table 4. The estimates of L 0,1
1,m and L 1,0
2,m show how many of the ESTs become
“shared” as the size of the additional sample increases. For instance, after m = 2000 additional
sequencing, we predict that in FRUIT 1 we will detect 144.67 ESTs originally observed only in
FRUIT 2. By comparing L 0,1
1,m and L 1,0
2,m it is apparent that the rate of detection for new values
specific to the FRUIT 1 sample in library FRUIT 2 is faster than vice versa. Also the number
of new genes not previously recorded in any of the two samples, L 0,0
1,m and L 0,0
2,m , is larger when
sampling additional genes for the FRUIT 2 library.
Citrus clementina: FRUIT 1 Citrus clementina: FRUIT 2
m ˆL 0,0
1,m ˆL 0,1
1,m ˆL 0
1,m ˆL 0
1,m –HPD ˆL 0,0
2,m ˆL 1,0
2,m ˆL 0
2,m L 0
2,m –HPD
200 67.65 14.45 82.09 (68, 97) 82.88 25.34 108.22 (93, 123)
400 135.28 28.89 164.17 (142, 186) 165.84 50.78 216.62 (193, 240)
600 202.84 43.37 246.20 (218, 275) 248.67 76.25 324.91 (294, 355)
800 270.52 57.86 328.38 (293, 364) 331.66 101.63 433.29 (396, 470)
1000 338.19 72.33 410.51 (369, 452) 414.57 127.08 541.65 (498, 585)
1200 405.82 86.80 492.61 (445, 540) 497.42 152.55 649.97 (600, 699)
1400 473.41 101.27 574.68 (521, 628) 580.21 177.97 758.19 (702, 814)
1600 541.03 115.76 656.79 (597, 716) 663.09 203.40 866.50 (805, 927)
1800 608.67 130.24 738.91 (675, 804) 745.92 228.88 974.80 (906, 1042)
2000 676.25 144.67 820.92 (750, 891) 828.90 254.24 1083.14 (1009, 1157)
Table 4: Citrus clementina: posterior expected number of new ESTs and 95% highest posterior
density intervals for the two libraries of fruits in the partially exchangeable framework for the
HPYP.
In accordance with Theorem 1, the posterior estimates of the quantities of interest turn out to
be linear in m. Thanks to Theorem 1 we also obtain estimates of the following one-step predictions
probabilities
P(X 1,n 1 +1 2 A 0,1 | X (n 1 )
1 , X (n 2 )
2 ) ⇡ 0.0723,
P(X 2,n 2 +1 2 A 1,0 | X (n 1 )
1 , X (n 2 )
2 ) ⇡ 0.127.
Hence, the slope of the linear estimator ˆL 0,1
2,m is higher than that of ˆL 1,0
1,m . This is also apparent
from Figure 1.
It is also interesting to compare Table 2 with Table 4. The appropriate quantities to focus
on are L 0
s,m from which the desired phenomenon of borrowing of strength is apparent. This is
13
(a) (b)
Figure 1: HPYP: total number of new ESTs ˆL 0
i,m in the exchangeable (a) and partially exchange-
able (b) settings as the size m of additional sample increases.
even more explicit in Figure 1, which depicts the posterior estimates of L 0
1,m and L 0
2,m as m
increases both in the exchangeable (Figure 1(a)) and partially exchangeable (Figure 1(b)) settings.
We may conclude that the discrepancies between ˆL 0
1,m and ˆL 0
2,m are much lower in the partially
exchangeable case. Furthermore the 95% HPD intervals are significantly narrower for the partial
exchangeable model, showing the beneficial influence of the borrowing of strength, which reduces
the uncertainty about the estimates. Besides, the estimates of the model parameters equal
(ˆ 0 , ˆ, ˆ✓, ˆ✓ 0 ) = (0.3449, 0.5595, 1241.40, 1044.54). (14)
Finally, we also consider the HDP case, which corresponds to  =  0 = 0. The estimated
quantities are reported in Table 5 and can be directly compared to those in Table 4 corresponding
to the HPYP. Figure 2 displays the posterior estimates of L 0
1,m and L 0
2,m , as m increases, for both
the exchangeable (Figure 2(a)) and partially exchangeable (Figure 2(b)) settings. The previous
considerations clearly apply also to the HDP. A first noteworthy, though not surprising, di↵erence
is that the rate of detection of new genes is much slower in the HDP case. Moreover, the shrinking
phenomenon in the partially exchangeable setup is less evident for the HDP. This means that,
besides the growth rate of new species being detected in additional samples, the key parameters
 and  0 have also a considerable e↵ect on the intensity of the shrinkage phenomenon. Finally,
one may argue as in Section 6.1 and note that there is no doubt about the HPYP yielding the
better performance: if the HDP were the model to use, the posterior estimates of  and  0 would
have been close to 0, namely consistent with the HDP model, which is clearly not the case as the
numerical values displyed in (14) illustrate.
14
Citrus clementina: FRUIT 1 Citrus clementina: FRUIT 2
m ˆL 0,0
1,m ˆL 0,1
1,m ˆL 0
1,m ˆL 0
1,m –HPD ˆL 0,0
2,m ˆL 1,0
2,m ˆL 0
2,m L 0
2,m –HPD
200 48.94 14.90 63.84 (51, 78) 62.54 26.83 89.37 (75, 104)
400 98.08 29.79 127.87 (108, 149) 125.10 53.66 178.76 (157, 200)
600 147.09 44.67 191.76 (166, 218) 187.46 80.53 267.00 (240, 296)
800 196.14 59.54 255.68 (225, 287) 249.81 107.34 357.17 (323, 391)
1000 245.13 74.41 319.55 (284, 356) 312.34 134.15 446.49 (407, 485)
1200 294.16 89.25 383.41 (344, 425) 374.73 160.99 535.71 (491, 580)
1400 343.22 104.04 447.26 (403, 494) 437.25 187.80 625.05 (575, 676)
1600 392.21 118.87 511.07 (462, 562) 499.90 214.54 714.43 (659, 770)
1800 441.33 133.72 575.05 (521, 631) 562.50 241.30 803.80 (744, 864)
2000 490.32 148.53 638.84 (581, 699) 625.06 268.09 893.14 (827, 958)
Table 5: Citrus clementina: posterior expected number of new ESTs and 95% highest posterior
density intervals for the two libraries of fruits in the partially exchangeable framework for the HDP.
(a) (b)
Figure 2: HDP: total number of new ESTs ˆL 0
i,m in the exchangeable (a) and partially exchangeable
(b) settings as the size m of additional sample increases.
References
[1] Barrientos, A.F., Jara, A., and Quintana, F.A. (2016). Fully nonparametric regression
for bounded data using dependent Bernstein polynomials. J. American Statist. Assoc., doi:
10.1080/01621459.2016.1180987.
[2] Bunge, J., Willis, A. and Walsh, F. (2014). Estimating the number of species in microbial
diversity studies. Annual Review of Statistics and Its Application 1, 427–445.
[3] Camerlenghi, F., Lijoi, A., Orbanz, P. and Pr ̈unster, I. (2016). Distribution theory
for hierarchical processes. Submitted.
15
[4] Carnap, R. (1950). Logical Foundations of Probability. University of Chicago Press, Chicago.
[5] Chao, A. and Jost, L. (2012). Coverage-based rarefaction and extrapolation: Standardizing
samples by completeness rather than size. Ecology 93, 2533–2547.
[6] Cifarelli, D. and Regazzini, E. (1978). Problemi statistici nonparametrici in condizioni
di scambiabilit`a parziale. Quaderni Istituto di Matematica Finanziaria, Universit`a di Torino.
[7] De Blasi, P., Favaro, S., Lijoi, A., Mena, R.H., Ruggiero, M. and Pr ̈unster, I.
(2015). Are Gibbs-type priors the most natural generalization of the Dirichlet process? IEEE
Trans. Pattern Anal. Mach. Intell. 37, 212–229.
[8] de Finetti, B. (1931). Probabilismo. Logos 14, 163–219. [Translated in Erkenntnis 31, 169–
223, 1989].
[9] de Finetti, B. (1937). La pr ́evision: ses lois logiques, ses sources subjectives. Ann. Inst. H.
Poincar ́e, 7, 1–68.
[10] de Finetti, B. (1938). Sur la condition d’ ́equivalence partielle. Actualit ́es scientifiques et
industrielles, 5–18.
[11] Gasthaus, J. and Teh, Y.W. (2010). Improvements to the sequence memoizer. Advances
in Neuro Information Processing Systems 23. 24th Annual Conference on Neural Informa-
tion Processing Systems 2010. Proceedings of a meeting held 6-9 December 2010, Vancouver,
British Columbia, Canada.
[12] Good, I.J. (1953). The population frequencies of species and the estimation of population
parameters. Biometrika, 40, 237–264.
[13] Good, I.J. and Toulmin, G.H. (1956). The number of new species, and the increase in
population coverage, when a sample is increased. Biometrika, 43, 45–63.
[14] Griffin, J.E. and Leisen, F. (2016), Compound random measures and their use in Bayesian
non-parametrics. J. R. Stat. Soc. B. doi:10.1111/rssb.12176
[15] Gutierrez, L., Mena, R.H. and Ruggiero, M. (2016). A time dependent Bayesian non-
parametric model for air quality analysis Computat. Statist. Data Anal. 95, 161–175.
[16] Hjort, N.L, Holmes, C., M ̈uller, P., Walker, S.G. (Eds.) (2010). Bayesian Nonpara-
metrics. Cambridge University Press, Cambridge, UK.
[17] Huynh, V., Phung, D., Venkatesh, S., Nguyen, X., Hoffman, M. and Bui, H.H.
(2016). Scalable nonparametric Bayesian multilevel clustering Proceedings of the ICML 2014.
[18] Jo, S., Lee, J., M ̈uller, P., Quintana, F.A., and Trippa, L. (2016). Dependent species
sampling models for spatial density estimation. Bayesian Anal., in press.
[19] Lijoi, A., Mena, R.H. and Pr ̈unster, I. (2007). Bayesian nonparametric estimation of the
probability of discovering a new species Biometrika, 94, 769–786.
[20] Lijoi, A, Mena, R.H. and Pr ̈unster, I. (2008). A Bayesian nonparametric approach for
comparing clustering structures in EST libraries. J. Comput. Biol., 15, 1315–1327.
16
[21] Lijoi, A. and Pr ̈unster, I. (2010). Models beyond the Dirichlet process. In Bayesian Non-
parametrics (Hjort, N.L., Holmes, C.C. M ̈uller, P., Walker, S.G. Eds.), pp. 80–136. Cambridge
University Press, Cambridge.
[22] MacEachern, S.N. (1999). Dependent nonparametric processes. In ASA Proceedings of the
Section on Bayesian Statistical Science. Alexandria: American Statistical Association, 50–55.
[23] MacEachern, S.N. (2000). Dependent Dirichlet processes. Technical Report. Department of
Statistics, Ohio State University.
[24] Mao, C.X. (2004). Prediction of the conditional probability of discovering a new class. J.
Amer. Statist. Assoc. 99, 1108–1118.
[25] Mena, R.H. and Ruggiero, M. (2016) Dynamic density estimation with di↵usive Dirichlet
mixtures Bernoulli 22, 901–926.
[26] M ̈uller, P., Quintana, F.A., Jara, A., Hanson, T. (2015). Bayesian Nonparametric
Data Analysis. Springer, New York.
[27] M ̈uller, P. and Quintana, F.A. (2004). Nonparametric Bayesian data analysis. Statist.
Science 19, 95–110.
[28] Nguyen, V., Phung, D., Nguyen, X., Venkatesh, S. and Bui, H.H. (2014). Bayesian
nonparametric multilevel clustering with group-level contexts. Proceedings of the ICML 2014.
[29] Nguyen, X. (2016). Borrowing strength in hierarchical Bayes: posterior concentration of the
Dirichlet base measure. Bernoulli 22, 1535–1571.
[30] Orlitsky, A., Suresh, A.T. and Wu, Y. (2016). Optimal prediction of the number of
unseen species. PNAS, 113, 13283–13288.
[31] Pitman, J. (2006). Combinatorial stochastic processes.  ́Ecole d’ ́et ́e de probabilit ́es de Saint-
Flour XXXII. Lecture Notes in Mathematics N. 1875, Springer, New York.
[32] Pitman, J. and Yor, M. (1997). The two-parameter Poisson-Dirichlet distribution derived
from a stable subordinator. Ann. Probab. 25, 855–900.
[33] Teh, Y.W. (2006). A hierarchical Bayesian language model based on Pitman–Yor processes.
In Proceedings of the 21st International Conference on Computational Linguistics and 44th
Annual Meeting of the Association for Computational Linguistics, 985–92. Morristown, NJ:
Association for Computational Linguistics.
[34] Teh, Y.W., Jordan, M.I., Beal, M.J. and Blei, D.M. (2006). Hierarchical Dirichlet
processes. J. Amer. Statist. Assoc. 101, 1566–1581.
[35] Vickers, J. (2011). The Problem of Induction. In The Stanford Encyclopedia of Philosophy
(Ed. E.N. Zalta).
[36] Zhu, W. and Leisen, F. (2015). A multivariate extension of a vector of two-parameter
Poisson-Dirichlet processes. J. Nonparam. Statist. 27, 89–105.