use strict;
use Statistics::Descriptive qw(:all);

my $file; my $ensemble_name; my $policy_name; my $eval_number; my $proportion; my $current_parameters;
my $temp_hash_to_string; my $stat; my $count; my $mean; my $min; my $max; my $var; my $stdev;

my @line; my @all_proportion_array;
my %count_proportion_hash; my %all_proportion_hash;

my $input_file = "banditPolicyProportionsVsEvalNumber.txt";
my $output_file = 'banditPolicyProportionsVsEvalNumberSummary.csv';
open INPUT, $input_file or die "Could not open '$input_file' for reading!";
open(OUTPUT, '>', $output_file) or die "Could not open file '$output_file'";

while (<INPUT>){
    @line = split /,/,$_;
    chomp(@line);
    $ensemble_name = $line[0];
    $policy_name = $line[1];
    $eval_number = $line[2];
    $proportion = $line[3];
    $current_parameters = "$ensemble_name,$policy_name,$eval_number";
    $count_proportion_hash{$current_parameters}++;
    if ($count_proportion_hash{$current_parameters} == 1){
        $all_proportion_hash{$current_parameters} = $proportion;
    }
    else {
        $all_proportion_hash{$current_parameters} = "$all_proportion_hash{$current_parameters},$proportion";
    }
}


print OUTPUT "Key,PolicyName,EvaluationNumber,NumberofProportionValues,MeanProportion,MinProportion,MaxProportion,VarProportion,StdevProportion\n";
foreach my $key1 (sort keys %all_proportion_hash){
	$temp_hash_to_string = "$all_proportion_hash{$key1}";
	@all_proportion_array = split /,/,$temp_hash_to_string;
	$stat = Statistics::Descriptive::Full->new();
	$stat->add_data(@all_proportion_array);
	$count = $stat->count();
	$mean = $stat->mean();
	$min = $stat->min();
	$max = $stat->max();
	$var  = $stat->variance();
	$var = sprintf("%.10f", $var);
	$stdev = $stat->standard_deviation();
	$stdev = sprintf("%.10f", $stdev);
	print OUTPUT "$key1,$count,$mean,$min,$max,$var,$stdev\n";
}
