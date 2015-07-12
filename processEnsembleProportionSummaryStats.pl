use strict;

my $ensemble; my $ensemble_name; my $policy_name; my $num_evals; my $num_runs;
my $mean_prop; my $min_prop; my $max_prop; my $var_prop; my $stdev_prop;

my @line;

my %mean_prop_hash; my %min_prop_hash; my %max_prop_hash; my %var_prop_hash; my %stdev_prop_hash;
my %all_policies_hash;

my $num_args = $#ARGV + 1;

if ($num_args != 1){
    print "Need to supply a single command line argument specifying the exact name of the ensemble algorithm that we want to break down.";
    exit;
}
else {
    $ensemble = $ARGV[0];
}
my $input_file_1 = "banditPolicyProportionsVsEvalNumberSummary.csv";
my $output_file_1 = "banditMeanPolicyProportionsVsEvalNumberSummary$ensemble.csv";
open INPUT1, $input_file_1 or die "Could not open input file 1 for reading!";

open OUTPUT1, '>', $output_file_1 or die "Could not open output file 1 for writing!";

# EnsembleName	PolicyName  EvaluationNumber	NumberofProportionValues	MeanProportion	MinProportion	MaxProportion	VarProportion	StdevProportion
#  EnsembleRandom	BinomialUCI	100	1	0.14	0.14	0.14	0	0
while (<INPUT1>){
	if ($. == 1){
		next;
	}
	@line = split /,/, $_;
	chomp(@line);
	$ensemble_name = $line[0];
	if ($ensemble ne $ensemble_name){
	    next;
	}
	$policy_name = $line[1];
	$num_evals = $line[2];
	$num_runs = $line[3];
	$mean_prop = $line[4];
	$min_prop = $line[5];
	$max_prop = $line[6];
	$var_prop = $line[7];
	$stdev_prop = $line[8];
	$all_policies_hash{$policy_name}++;
	#if ($data_points >= 10) {
	#if ($data_points >= 23) {}
        $mean_prop_hash{$num_evals}{$policy_name} = $mean_prop;
        $min_prop_hash{$num_evals}{$policy_name} = $min_prop;
        $max_prop_hash{$num_evals}{$policy_name} = $max_prop;
        $var_prop_hash{$num_evals}{$policy_name} = $var_prop;
        $stdev_prop_hash{$num_evals}{$policy_name} = $stdev_prop;
	#}
}

# Print out header row
print OUTPUT1 "EvalNumber,";
foreach my $key (keys %all_policies_hash) {
	print OUTPUT1 "$key,";
}
print OUTPUT1 "\n";

foreach my $key1 (sort {$a<=>$b} keys %mean_prop_hash){
	print OUTPUT1 "$key1,";
    foreach my $key2 (keys %all_policies_hash){
        print OUTPUT1 "$mean_prop_hash{$key1}{$key2}," ;
    }
    print OUTPUT1 "\n";
}

