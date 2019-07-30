def proceed(squelch=False):
  if squelch:
    return True
  inp=input('Proceed? (y/[n]):  ')
  if inp.lower() in ('y','yes','true'):
    return True
  else:
    return False
