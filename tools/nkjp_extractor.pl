#!/usr/bin/perl

use strict;
use warnings;
use File::Find;
use XML::Simple;

die "Usage: $0 root_dir output_dir\n" unless @ARGV == 2;
my ($root_dir, $output_dir) = @ARGV;


find(\&process_file, $root_dir);

sub process_file {
    my $file = $_;

    if (-f $file && $file =~ /text\.xml$/) {
        my @directories = split('/', $File::Find::dir);
        my $output_file = pop @directories;
        $output_file .= ".txt";
        $output_file = "$output_dir/" . join('/', @directories[0]) . "/$output_file";
        
        open(my $fh, '<', $file) or die "Could not open file '$file' $!";
        my $content = do { local $/; <$fh> };
        close($fh);

        my $extracted_cont = '';

        while ($content =~ /<ab.*>(.*?)<\/ab>/g) {
            my $extracted = $1;
            $extracted_cont = $extracted_cont."\n".$extracted;
        }
        
        open(my $fh, '>', $output_file) or die "Could not open file '$output_file' $!";
        print $fh $extracted_cont;
        close $fh;
    }
}