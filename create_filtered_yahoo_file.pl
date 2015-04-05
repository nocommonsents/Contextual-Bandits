use strict;
use warnings;

open (FILE, "./exploChallenge/ydata-fp-td-clicks-v2_0.20111002-08.txt");
open OUTPUT, ">./exploChallenge/ydata-fp-td-clicks-v2_0.20111002-08-filtered10percent.txt";

my $feature_vector;
my @all_features;
my @features_to_keep = (1,13,14,15,18,16,17,19,23,29,25,27,22,21,24,26,30,45,35,12,32,11,36,20,44,33,43,40,39);
my %features_to_keep_hash = map { $_ => 1 } @features_to_keep;

while (<FILE>){
# Sample line: 1317513292 id-552077 0 |user 1 7 11 37 13 23 16 18 17 35 15 14 30 20 |id-552077 |id-555224 |id-555528 |id-559744 |id-559855 |id-560290 |id-560518 |id-560620 |id-563115 |id-563582 |id-563643 |id-563787 |id-563846 |id-563938 |id-564335 |id-564418 |id-564604 |id-565364 |id-565479 |id-565515 |id-565533 |id-565561 |id-565589 |id-565648 |id-565747 |id-565822
# Key: Time id-articlePicked binaryClick |user featureVector all articles not picked	
	if ($_ =~ /(\d+ id-\d+ \d+ \|user )([\d\s]+) (\|[\w\d\-\s\|]+)/){
		if ($. % 100000 == 0) {
			print "Currently processing line $. of input file.\n";
		}
		print OUTPUT "$1 ";
		$feature_vector = $2;
		@all_features = split /\s/, $feature_vector;
		#print "@all_features\n";
		for my $element (@all_features) {
			$element =~ s/^\s+|[\s+\n]$//g;
			if (exists($features_to_keep_hash{$element})) {
				print OUTPUT "$element ";
				#print "$element ";
			}
		}
		print OUTPUT "$3";
	}
}
