Bringing new medical products to patients is a long and complex process. Once a promising compound has been discovered, and its mechanisms of action and potential efficacy are well understood, it is moved to the clinical trial phase. These phases are split as follows:

Phase 1 determines if a drug is safe to use on humans, by targeting multiple dosing regimens. Around 50 patients are in these trials.
Phase 2 tests the efficacy of therapeutic doses on patients. The goal is to assess if the drug has any efficacy, and its potential side effects. This usually involves a couple of hundred patients.
Phase 3 determines the effectiveness of the drug at therapeutic doses on a couple of thousands of patients.
The drug can be deployed on the market only after a successful phase 3, after which it will be continuously monitored for long-term side effects. Most drug candidates never make it to the end of Phase 3.

Quantum computers are expected to improve many parts of the pharmaceutical pipeline. They are especially promising for one of the most crucial parts of the pharmaceutical pipeline: drug discovery. Quantum chemistry, protein-ligand docking, and other types of simulations are expected to benefit from upcoming quantum computers, leading to better drug candidates.

However, there is another area where quantum computing may be helpful. While relying on a small amount of data, this area must answer some of the most important and difficult questions: What should be the dosage when switching from animals to human testing? What dosage should participants take in Phase 2 trials, where efficacy is assessed? If the dosage is too low, efficacy will be limited. If the dosage is too high, side effects may become predominant. The scientific field in charge of answering these questions is called pharmacokinetics and pharmacodynamics (PK/PD).


Clinical R&D Timeline
Pre-clinical
Several years
Laboratory Studies
Generate experimental data about solubilities, solvents selection, dosing and toxicity levels via animal experiments or organ chips.
Phase 1
Several months
Safety
Evaluate safety with first human test group by gathering information how the drug formulation interacts with the human body. ~50 patients
Phase 2
Several months
Safety & Dosing
Further safety evaluations. Monitoring side effects. Check which dose works best and is most effective. ~200 patients
Phase 3
Several years
Safety & Efficacy
Effectiveness confirmation and continued monitoring. ~1,000+ patients
FDA Review
~1 year
Regulatory Approval
Comprehensive review of all clinical data before market approval. Continuous monitoring for long-term side effects post-approval.

Pharmacokinetics-pharmacodynamics

Pharmacokinetics (PK) and pharmacodynamics (PD) are two related concepts that are used to understand and predict how medicines work in the body:

Pharmacokinetics (PK): describes how the body interacts with the drug in terms of its absorption, distribution, metabolism, and elimination. It models how drug concentration changes over time in the body.
Pharmacodynamics (PD): describes how the drug interacts with the body, linking drug concentration to its biological effect. PD models relate the concentration of a drug to the intensity and duration of its effects.
PK/PD: A combined PK/PD model integrates both aspects; it predicts the time course of drug effects based on the dosing regimen.
PK/PD modeling is generally based on ordinary differential equations (ODEs) to describe both PK and PD models. Existing models are by essence meant to be empirical and interpretable, while providing sufficient variability to simulate and predict drugs behavior under different conditions. These modeling strategies have been used for around 30 years.

**Alternative Methods**

This mechanistic approach has proven effective, and ODE-based models have become prominent in PK/PD. However, this also makes PK/PD labor intensive: they require extensive manual effort, expert knowledge, and many trial-and-errors. Because of the small number of participants in early clinical trial phases, PK/PD modeling is constraint by small sample sizes, limited time points available, and only a few dosing regimens investigated, which limits their accuracy and applicability. Today PK/PD is mechanistic, and hypothesis driven. Several directions have been explored to extend PK/PD. Most attempts at making it more data-driven are based on machine learning (ML), whether through ML techniques [1,2], deep learning[3-7], reinforcement learning [8], agent-based modeling [9.10], and more [1]. In 2021, the FDA mentioned “a new approach to pharmacometrics”, citing a paper showing encouraging results using an LSTM architecture [3]. The method, however, failed at predicting the PD response of drug dosing regimen different than the one used for training. These data-driven approaches have the potential to reduce human supervision and time needed when it comes to model development. They could also help discover new relationships without prior mechanistic knowledge and may be more flexible with more diverse types of data types and covariates.

However, many data-driven approaches are limited by the scarcity of available PK/PD data, which reduces their applicability.


**Innovation Challenge: Quantum Computing for PK/PD**

Your challenge, if you wish to accept it, will shine some light on the relevancy of quantum computing methods for PK/PD related problems. Because it is crucial to the pharmaceutical pipeline, and since it relies on early phase clinical trial data, PK/PD could prove to be an interesting and valuable playground for a variety of quantum computing methods.

Quantum machine learning (QML) methods could be a relevant example [11]. A growing body of literature heuristically explores if access to a quantum computer can improve learning algorithms [12-22]. One of the most promising features of quantum machine learning are their expressivity[23-29], trainability[30-39], and generalization[40-55]. This generalization aspect is especially interesting, since QML may help generalize PK/PD models better despite few training data.

Other applications could be within the standard PK/PD modeling process, as it includes differential equations, parameter estimation, bootstrap methods, and other techniques where quantum computing methods may help enhance accuracy and reduce computation time.

No matter the direction taken, creativity is most welcomed, provided that scientific rigor remains the leading principle. The ultimate measure of success lies in the tangible value these quantum computing based solutions can provide to PK/PD.

**Challenge subject**

A phase 1 placebo-controlled clinical trial testing three dose levels (1 mg, 3 mg, and 10 mg) of a new compound dosed once daily for three weeks has just been completed. Data from 48 subjects is available (36 on active treatment and 12 on placebo). The subjects weigh between 50 and 100 kg and half of them have also taken some concomitant medication, i.e., another drug. Pharmacokinetic (PK) data (measurements of compound concentration) is available for the subjects randomized to active treatment, and pharmacodynamic (PD) data (measurement of a biomarker) is available for all subjects. Prior knowledge about the structure of a PK-PD model that can describe the observed data is limited, but the data shows that the new compound has a long half-life and that it can suppress the level of the biomarker. The data also shows that the concomitant medication affects the biomarker.

Disclaimer: This dataset is synthetic and has been generated to reflect realistic clinical trial data. The data file (EstData.csv) is located in the data folder of the challenge repository:

Download Dataset here: https://github.com/Quantum-Innovation-Challenge/LSQI-Challenge-2025/tree/main/data

**Tasks to solve**

Your task is to develop a quantum-enhanced model to answer the following questions:

What is the daily dose level (in whole multiples of 0.5 mg) that ensures that 90% of all subjects in a population similar to the one studied in the phase 1 trial achieve suppression of the biomarker below a clinically relevant threshold (3.3 ng/mL) throughout a 24-hour dosing interval at steady-state?

Which weekly dose level (in whole multiples of 5 mg) has the same effect over a 168-hour dosing interval at steady-state, if the compound was dosed once-weekly?

Suppose we change the body weight distribution of the population to be treated to 70-140 kg, how does that affect the optimal once-daily and once-weekly doses?

Suppose we impose the restriction that concomitant medication is not allowed. How does that affect the optimal once-daily and once-weekly doses?

How much lower would the optimal doses in the above scenarios be if we were to ensure that only 75% of all subjects achieve suppression of the biomarker below the clinically relevant threshold (3.3 ng/mL)

Note: We will not discriminate based on the type of algorithm and on which infrastructure or processing unit they are run (CPUs, GPUs, QPUs, etc.). This means quantum-inspired methods (e.g., tensor network-based), along with methods from digital or analog quantum computing, quantum annealing, boson sampling, and others are welcome. What matters most are the performance, scalability, and generalizability of the developed methods, along with how early they can demonstrate benefits. The developed method will be assessed by the committee by applying it on new PK/PD datasets.

**Your impact**

Your work will bring many new insights on how to improve and expand current PK/PD modelling methods, and on the potential of quantum computing for related applications.

Standard way of tackling this in PK/PD

A skilled PK/PD modeling scientist could answer these questions within 2 days. The standard way of solving the task involves multiple sequential steps:

Determine an appropriate model structure for PK (one- or two-compartment model, linear versus nonlinear absorption, linear versus nonlinear elimination…etc.)
Assess how to best describe the PK variability between subjects using the observed compound concentration data
Determine an appropriate model structure for PD (linear versus nonlinear response, direct versus indirect response model…etc.)
Assess how to best describe the PD variability between subjects
Develop a joint PK/PD model
Use this PK/PD model to do simulations and help determine what the dose level should be.
Model estimation would be done using a nonlinear mixed effects model estimation algorithm (e.g. using the packages [NONMEM](https://www.nonmem.org/) or [NLMIXR](https://cran.r-project.org/web/packages/nlmixr2/index.html)).

**Dataset explanations and metadata**

Dataset shape: 2820 rows x 11 columns

Dataset columns:

ID: Subject identifier.

BW: Body weight of the subject in kg.
COMED: Concomitant medication indicator for the subject (0: No, 1: Yes)
DOSE: Dose level in mg. This column represents the amount of drug administered to the subject. It is typically measured in units like milligrams (mg) or micrograms (μg). The dose is a critical factor in determining the drug’s pharmacokinetic and pharmacodynamic properties.
TIME: Time in hours. Indicates the time elapsed since the start of the first drug administration. Time is typically measured in hours or minutes and is essential for plotting concentration-time profiles.
DV (Dependent Variable): Compound concentration (mg/L) for DVID=1. Biomarker level (ng/mL) for DVID=2. This column usually represents observed data, such as drug concentration in plasma or another biological matrix. It can also refer to the measurement of a biomarker or response variable affected by the drug, such as blood pressure or heart rate.
EVID (Event ID): This is an event identifier used in NONMEM (a common software in pharmacometrics). It signifies the type of event occurring:
EVID = 0 for observation events (e.g., a concentration measurement),
EVID = 1 for dosing events.
MDV (Missing Dependent Variable): This value indicates whether the dependent variable (DV) is missing. An MDV of 1 means the DV value is missing, while 0 means it is present.
AMT (Amount): Stands for the actual dose amount administered, especially applicable during infusion dosing. AMT will be zero for observation records.
CMT (Compartment): This denotes the compartment where the event (e.g., dosing or sampling) occurs. In PK models, different compartments (like central and peripheral) help to describe drug kinetics.
DVID (Dependent Variable ID): This identifier helps distinguish between different types of DVs in the dataset. For example, you might have multiple measurement types such as concentrations and biomarkers.