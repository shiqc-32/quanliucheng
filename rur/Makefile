OBJS =	Srp_debug.o \
         Srf_profile.o \
			Sfind_sort_largest.o \
			Sreconstructed_layers.o \
			Spre_dz0.o \
			Sdz0.o \
			RealUrbanR.o

CMD = RealUrbanR.x 

RM = \rm -f

F90 = gfortran

#
# gfortran
#
ifeq ($(F90),gfortran)
	FFLAGS = -ffree-form 
endif

#
# ifx
#
ifeq ($(F90),ifort)
	FFLAGS = -free
endif

$(CMD):	$(OBJS)
	$(F90) $(FFLAGS) -o $(@) $(OBJS) $(NETCDFLIB)

.SUFFIXES:
.SUFFIXES: .o .f90 .f

%.o: %.f
	$(F90) $(FFLAGS) -c $< $(NETCDFINC)

%.o: %.f90
	$(F90) $(FFLAGS) -c $< $(NETCDFINC)

clean:
	$(RM) $(OBJS) $(CMD) *.mod *~

