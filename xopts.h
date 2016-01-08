/* getopts generation code using xmacros */
#ifndef XOPTS_H
#define XOPTS_H 1
#include <getopt.h>

#define X(LONG, REQARG, SHORT, T, DFLT, PARSE) T LONG;
struct {
  GET_OPTS
#undef X
#define X(LONG, REQARG, SHORT, T, DFLT, PARSE) DFLT,
} options = {
  GET_OPTS
};
#undef X

static char*
gen_optstr()
{
  static char s[64];
  int i = 0;
  s[i++] = ':';

#define X(LONG, REQARG, SHORT, T, DFLT, PARSE) \
  s[i++] = SHORT; \
  if (REQARG == required_argument) s[i++] = ':';
    GET_OPTS
#undef X

  s[i] = '\0';
  return s;
}

static void
parse_options(int argc, char *argv[])
{
  int c;
  
#define X(LONG, REQARG, SHORT, T, DFLT, PARSE) {#LONG, REQARG, NULL, SHORT},
  static struct option longopts[] = {
    GET_OPTS
    {NULL, 0, NULL, 0}
  };
#undef X

  while ((c = getopt_long(argc, argv, gen_optstr(), longopts, NULL)) != -1){
  switch(c){
  #define X(LONG, REQARG, SHORT, T, DFLT, PARSE) \
    case SHORT: options.LONG = PARSE; break;
      GET_OPTS
    default: opterr = 1; break;
  }}
  #undef X  
}

static void
dump_options()
{
#define X(LONG, REQARG, SHORT, T, DFLT, PARSE) printf("  -%c            --%-12s\n", SHORT, #LONG);
    GET_OPTS
#undef X  
}



#endif
