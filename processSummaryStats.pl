
my $input_file = "banditCtrOutputSummary.csv";
my $output_file_1 = "banditMeanAERSummary.csv";
my $output_file_2 = "banditMinAERSummary.csv";
my $output_file_3 = "banditMaxAERSummary.csv";
my $output_file_4 = "banditVarAERSummary.csv";
my $output_file_5 = "banditStDevAERSummary.csv";

my $policy; my $dataset; my $num_evals; my $num_runs; my $data_points;
my $mean_aer; my $min_aer; my $max_aer; my $var_aer; my $stdev_aer;

my @line;

my %mean_aer_hash; my %min_aer_hash; my %max_aer_hash; my %var_aer_hash; my %stdev_aer_hash;
my %all_policies_hash;

open INPUT, $input_file or die "Could not open input file for reading!";
open OUTPUT1, '>', $output_file_1 or die "Could not open output file 1 for writing!";
open OUTPUT2, '>', $output_file_2 or die "Could not open output file 2 for writing!";
open OUTPUT3, '>', $output_file_3 or die "Could not open output file 3 for writing!";
open OUTPUT4, '>', $output_file_4 or die "Could not open output file 4 for writing!";
open OUTPUT5, '>', $output_file_5 or die "Could not open output file 5 for writing!";

##
# NumEvals Policy_1 Policy_2 Policy_3... (Mean AER)
# 100  0.037 0.054 0.052 0.084 0.064

#Policy, Dataset, NumEvals, NumRuns, MeanAER, MinAER, MaxAER, VarAER, StDevAER
#eAnnealing	y	100	28	0.037857143	0.01	0.07	0.000306349	0.017502834
while (<INPUT>){
	if ($. == 1){
		next;
	}
	@line = split /,/, $_;
	chomp(@line);
	$policy = $line[0];
	$num_evals = $line[2];
	$data_points = $line[3];
	$mean_aer = $line[4];
	$min_aer = $line[5];
	$max_aer = $line[6];
	$var_aer = $line[7];
	$stdev_aer = $line[8];
	$all_policies_hash{$policy}++;
	if ($data_points >= 23 || $policy=~/^UCB1/) {
		$mean_aer_hash{$num_evals}{$policy} = $mean_aer;
		$min_aer_hash{$num_evals}{$policy} = $min_aer;
		$max_aer_hash{$num_evals}{$policy} = $max_aer;
		$var_aer_hash{$num_evals}{$policy} = $var_aer;
		$stdev_aer_hash{$num_evals}{$policy} = $stdev_aer;
	}
	
}
print OUTPUT1 "t,";
print OUTPUT2 "t,";
print OUTPUT3 "t,";
print OUTPUT4 "t,";
print OUTPUT5 "t,";
foreach my $key (keys %all_policies_hash) {
	print OUTPUT1 "$key,";
	print OUTPUT2 "$key,";
	print OUTPUT3 "$key,";
	print OUTPUT4 "$key,";
	print OUTPUT5 "$key,";
}
print OUTPUT1 "\n";
print OUTPUT2 "\n";
print OUTPUT3 "\n";
print OUTPUT4 "\n";
print OUTPUT5 "\n";

foreach my $key1 (sort {$a<=>$b} keys %mean_aer_hash){
	print OUTPUT1 "$key1,";
    foreach my $key2 (keys %all_policies_hash){
        print OUTPUT1 "$mean_aer_hash{$key1}{$key2}," ;
    }
    print OUTPUT1 "\n";
}
foreach my $key3 (sort {$a<=>$b} keys %min_aer_hash){
	print OUTPUT2 "$key3,";
    foreach my $key4 (keys %all_policies_hash){
        print OUTPUT2 "$min_aer_hash{$key3}{$key4}," ;
    }
    print OUTPUT2 "\n";
}
foreach my $key5 (sort {$a<=>$b} keys %max_aer_hash){
	print OUTPUT3 "$key5,";
    foreach my $key6 (keys %all_policies_hash){
        print OUTPUT3 "$max_aer_hash{$key5}{$key6}," ;
    }
    print OUTPUT3 "\n";
}
foreach my $key7 (sort {$a<=>$b} keys %var_aer_hash){
	print OUTPUT4 "$key7,";
    foreach my $key8 (keys %all_policies_hash){
        print OUTPUT4 "$var_aer_hash{$key7}{$key8}," ;
    }
    print OUTPUT4 "\n";
}
foreach my $key9 (sort {$a<=>$b} keys %stdev_aer_hash){
	print OUTPUT5 "$key9,";
    foreach my $key10 (keys %all_policies_hash){
        print OUTPUT5 "$stdev_aer_hash{$key9}{$key10}," ;
    }
    print OUTPUT5 "\n";
}
