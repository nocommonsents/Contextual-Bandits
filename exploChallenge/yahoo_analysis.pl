use strict;
use warnings;

my $average_time_between_log_lines;	
my $start_time; my $latest_time; my $total_time;
my $recommended_article; my $binary_total = 0; my $click_rate;
my $full_candidate_article_string; my $candidate_article_count;
my $total_candidate_articles; my $average_candidate_articles;
my $key1; my $unique_candidate_articles_count; my $unique_feature_vectors_count;

my @article_id_matches;

my %temp_candidate_articles;
my %unique_candidate_articles;
my %unique_feature_vectors; 

open FILE, "ydata-fp-td-clicks-v2_0.20111002-all" or DIE $!;

while (<FILE>){
# Sample line: 1317513292 id-552077 0 |user 1 7 11 37 13 23 16 18 17 35 15 14 30 20 |id-552077 |id-555224 |id-555528 |id-559744 |id-559855 |id-560290 |id-560518 |id-560620 |id-563115 |id-563582 |id-563643 |id-563787 |id-563846 |id-563938 |id-564335 |id-564418 |id-564604 |id-565364 |id-565479 |id-565515 |id-565533 |id-565561 |id-565589 |id-565648 |id-565747 |id-565822
# Key: Time id-articlePicked binaryClick |user featureVector all articles not picked	
	if ($_ =~ /(\d+) (id-\d+) (\d+) \|user ([\d\s]+) \|([\w\d\-\s\|]+)/){
		if ($. % 100000 == 0) {
			print "Currently processing line $. of input file.\n";
		}
		if ($. == 1) {
			$start_time = $1;
		}
		else {
			$latest_time = $1;
		}
		$recommended_article = $2;
		$binary_total += $3;
		if (not exists $unique_feature_vectors{$4}){
			$unique_feature_vectors{$4} = 1;
		}
		$full_candidate_article_string = $5;
		@article_id_matches = $full_candidate_article_string =~ m/(\d+)/g;
		$temp_candidate_articles{$_}++ for (@article_id_matches);
		foreach $key1 (keys %temp_candidate_articles) {
			if (not exists $unique_candidate_articles{$key1}){
				$unique_candidate_articles{$key1} = 1;
			}
		}
    	$candidate_article_count = scalar(@article_id_matches);
		$total_candidate_articles+=$candidate_article_count;
		$candidate_article_count = 0;
	}
}
$total_time = $latest_time - $start_time;
$average_time_between_log_lines = $total_time/$.;
$click_rate = $binary_total / $.;
$unique_feature_vectors_count = keys %unique_feature_vectors;
$unique_candidate_articles_count = keys %unique_candidate_articles;
$average_candidate_articles = $total_candidate_articles / $.;
print "Elapsed time: $total_time seconds.\n";
print "Average time between log lines: $average_time_between_log_lines.\n";
print "Number of lines: $.\n";
print "Click rate: $click_rate. \n";
print "Number of unique feature vectors: $unique_feature_vectors_count\n";
print "Number of unique candidate_articles: $unique_candidate_articles_count\n";
print "Average number of candidate articles: $average_candidate_articles\n";	
