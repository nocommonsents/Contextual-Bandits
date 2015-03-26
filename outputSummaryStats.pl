use strict;

use Cwd qw();
use File::Slurp;
use Statistics::Descriptive qw(:all);

my $path = Cwd::cwd();
my @files = read_dir $path;
my @files_to_process = $full_candidate_article_string =~ m/(\d+)/g;

my $key1; my $key2;
my $current_policy; my $current_parameters; my $std_dev; my $temp_hash_to_string;
my $stat; my $count; my $mean; my $min; my $max; my $var; my $stdev;
my $converted_runtime; my $time_bin;

my @ctr_array; my @runtime_array; my @time_vs_ctr_array;

my %total_ctr_hash; my %count_ctr_hash;
my %all_ctr_hash;
my %total_runtime_hash; my %count_runtime_hash;
my %all_runtime_hash;
my %total_time_vs_ctr_hash; my %count_time_vs_ctr_hash;
my %all_time_vs_ctr_hash;

my $ctr_filename = 'banditCtrOutputSummary.csv';
open(OUTPUT, '>', $ctr_filename) or die "Could not open file '$ctr_filename'";
my $runtime_filename = 'banditRuntimeOutputSummary.csv';
open(OUTPUT2, '>', $runtime_filename) or die "Could not open file '$runtime_filename'";
my $ctr_vs_time_filename = 'banditCtrVsTimeOutputSummary.csv';
open(OUTPUT3, '>', $ctr_vs_time_filename) or die "Could not open file '$ctr_vs_time_filename'";

for my $file (@files) {
	if ($file =~ /^banditOutputs/){
		#print "$file\n";
		open FILE, $file or DIE $!;
		while (<FILE>){
			if ($_ =~ /Policy: ([\w\d\.]+)/){
				# Keep track of policy being used
				$current_policy = $1;
			}
			# Sample line: eGreedy0.1,y,100,0.03,0.634

			elsif ($_ =~ /([\w\d\.]+),(\w+),(\d+),([\d\.]+),([\d\.]+)/){
				$current_parameters = "$1,$2,$3";
				# Bin times into 10 second buckets
				$time_bin = int($5/10)*10;
				$time_parameters = "$1,$2,$time_bin";
				
				$total_ctr_hash{$current_parameters} += $4;
				$count_ctr_hash{$current_parameters}++;
				if ($count_ctr_hash{$current_parameters} == 1){
					$all_ctr_hash{$current_parameters} = $4;
				}
				else {
					$all_ctr_hash{$current_parameters} = "$all_ctr_hash{$current_parameters},$4";
				}
				
				#print $_;
				$total_time_vs_ctr_hash{$time_parameters} += $4;
				$count_time_vs_ctr_hash{$time_parameters}++;
				if ($count_time_vs_ctr_hash{$time_parameters} == 1) {
					$all_time_vs_ctr_hash{$time_parameters} = $4;
				}
				else {
					$all_time_vs_ctr_hash{$time_parameters} = "$all_time_vs_ctr_hash{$time_parameters},$4";
				}
			}
			# Sample line: Total runtime: 2296449 ms
			elsif ($_ =~ /Total runtime: (\d+) ms/){
				# Convert from sec to msec
				$converted_runtime = $1/1000;
				$total_runtime_hash{$current_policy} += $converted_runtime;
				$count_runtime_hash{$current_policy} ++;
				if ($count_runtime_hash{$current_policy} == 1){
					$all_runtime_hash{$current_policy} = $converted_runtime;
				}
				else {
					$all_runtime_hash{$current_policy} = "$all_runtime_hash{$current_policy},$converted_runtime";
				}
			}
		}
	}
}
print OUTPUT "Key,,,NumberofCTRValues,MeanCTR,MinCTR,MaxCTR,VarCTR,StdevCTR\n";
foreach $key1 (sort keys %total_ctr_hash){
	$temp_hash_to_string = "$all_ctr_hash{$key1}";
	@ctr_array = split /,/,$temp_hash_to_string;
	$stat = Statistics::Descriptive::Full->new();
	$stat->add_data(@ctr_array); 
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
print OUTPUT2 "Key,,,NumberofRuntimeValues,MeanRuntime,MinRuntime,MaxRuntime,VarRuntime,StdevRuntime,AllRuntimes\n";
foreach $key2 (sort keys %total_runtime_hash){
	$temp_hash_to_string = "$all_runtime_hash{$key2}";
	@runtime_array = split /,/,$temp_hash_to_string;
	$stat = Statistics::Descriptive::Full->new();
	$stat->add_data(@runtime_array); 
	$count = $stat->count();
	$mean = $stat->mean();
	$min = $stat->min();
	$max = $stat->max();
	$var  = $stat->variance();
	$var = sprintf("%.2f", $var);
	$stdev = $stat->standard_deviation();
	$stdev = sprintf("%.2f", $stdev);
	print OUTPUT2 "$key2,$count,$mean,$min,$max,$var,$stdev,$temp_hash_to_string\n";
}

print OUTPUT3 "Key,,,NumberofTimeVsCTRValues,MeanCTR,MinCTR,MaxCTR,VarCTR,StdevCTR,AllCTRs\n";
foreach $key3 (sort keys %total_time_vs_ctr_hash){
	$temp_hash_to_string = "$all_time_vs_ctr_hash{$key3}";
	@time_vs_ctr_array = split /,/,$temp_hash_to_string;
	$stat = Statistics::Descriptive::Full->new();
	$stat->add_data(@time_vs_ctr_array); 
	$count = $stat->count();
	$mean = $stat->mean();
	$min = $stat->min();
	$max = $stat->max();
	$var  = $stat->variance();
	$var = sprintf("%.2f", $var);
	$stdev = $stat->standard_deviation();
	$stdev = sprintf("%.2f", $stdev);
	print OUTPUT3 "$key3,$count,$mean,$min,$max,$var,$stdev,$temp_hash_to_string\n";
}
