use strict;
use warnings;

my $feature_vector; my $line_number; my $proportion;
my @all_features;
my %feature_hash;

open FILE, "./exploChallenge/ydata-fp-td-clicks-v2_0.20111002-08.txt" or DIE $!;
open OUTPUT, ">./feature_support_output.txt" or DIE $!;

while (<FILE>){
# Sample line: 1317513292 id-552077 0 |user 1 7 11 37 13 23 16 18 17 35 15 14 30 20 |id-552077 |id-555224 |id-555528 |id-559744 |id-559855 |id-560290 |id-560518 |id-560620 |id-563115 |id-563582 |id-563643 |id-563787 |id-563846 |id-563938 |id-564335 |id-564418 |id-564604 |id-565364 |id-565479 |id-565515 |id-565533 |id-565561 |id-565589 |id-565648 |id-565747 |id-565822
# Key: Time id-articlePicked binaryClick |user featureVector all articles not picked	
	if ($_ =~ /\d+ id-\d+ \d+ \|user ([\d\s]+) \|([\w\d\-\s\|]+)/){
		if ($. % 100000 == 0) {
			print "Currently processing line $. of input file.\n";
		}
		$feature_vector = $1;
		@all_features = split /\s/, $feature_vector;
		for my $element (@all_features) {
			$feature_hash{$element}++;
		}
		$line_number = $.;
	}
}

foreach my $key1 (sort {$feature_hash{$b}<=>$feature_hash{$a}} keys  %feature_hash){
	$proportion = $feature_hash{$key1}/$line_number;
	print "$key1\t$feature_hash{$key1}\t$proportion\n";
	print OUTPUT "$key1\t$feature_hash{$key1}\t$proportion\n";
}
