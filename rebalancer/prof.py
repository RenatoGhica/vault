import pstats
p=pstats.Stats('prof.txt')
p.strip_dirs().sort_stats(-1).print_stats()
p.sort_stats('cumulative').print_stats(10)
p.sort_stats('time').print_stats(10)
p.sort_stats('time', 'cum').print_stats(.5, 'init')
p.print_callers(.5)



