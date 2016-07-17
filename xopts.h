/* getopts generation code using xmacros 
    N, REQ, F, T, DFLT, FUNC, DESC
*/
#ifndef XOPTS_H
#define XOPTS_H 1
#include <getopt.h>

#define X(N, REQ, F, T, DFLT, FUNC, DESC) T N;
struct {
  GET_OPTS
#undef X
#define X(N, REQ, F, T, DFLT, FUNC, DESC) DFLT,
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

#define X(N, REQ, F, T, DFLT, FUNC, DESC) \
  s[i++] = F; \
  if (REQ == required_argument) s[i++] = ':';
    GET_OPTS
#undef X

  s[i] = '\0';
  return s;
}

static void
parse_options(int argc, char *argv[])
{
  int c;
  
#define X(N, REQ, F, T, DFLT, FUNC, DESC) {#N, REQ, NULL, F},
  static struct option longopts[] = {
    GET_OPTS
    {NULL, 0, NULL, 0}
  };
#undef X

  while ((c = getopt_long(argc, argv, gen_optstr(), longopts, NULL)) != -1){
  switch(c){
  #define X(N, REQ, F, T, DFLT, FUNC, DESC) \
    case F: options.N = FUNC; break;
      GET_OPTS
    default: opterr = 1; break;
  }}
  #undef X  
}

static void
dump_options()
{
#define X(N, REQ, F, T, DFLT, FUNC, DESC) \
  printf("  -%c  --%s  %s\n", F, #N, DESC);
    GET_OPTS
#undef X  
}



#endif
