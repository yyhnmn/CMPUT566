Yuhan Ye 1463504
Cmput 566 A3

All functions implemented

CHANGE
    parser.add_argument('--dataset', type=str, default="susy",
                        help='Specify the name of the dataset')
TO
    parser.add_argument('--dataset', type=str, default="census",
                        help='Specify the name of the dataset')
to change dataset from susy to census and vice versa

stratifiedkfold implemented below cross_validation
CHANGE
             best_parameters[learnername] = cross_validate(
                 10, Xtrain, Ytrain, Learner, params)
TO
            # # stratified kfold test            
            best_parameters[learnername] = stratifiedKfold(
                10, Xtrain, Ytrain, Learner, params)
to use stratifiedkfold, and vice versa

stratifiedkfold not works for census dataset
kfold cross_validation and stratifiedkfold cross_validation for numruns times
Takes long time for training so be patient

Kernal logistic regression has very high variance, the error fluctuation range is very large
