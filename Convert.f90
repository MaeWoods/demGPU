program convert

	implicit none

	integer :: ios	
	integer :: np	
	integer :: counter	
	integer :: i	
	
	logical :: eof
	
	character(100) :: pvdstring
	character(20) :: timestring
	character(20) :: filestring	
	character(20) :: vtustring
	character(20) :: npstring
	
	real(8) :: time
	real(8),dimension(:),allocatable :: radius
	real(8),dimension(:,:),allocatable :: position
	real(8),dimension(:,:),allocatable :: velocity
	real(8),dimension(:,:),allocatable :: alignment
	
	open(1,file="particle_data.dat",status="old",form="formatted",action="read",iostat=ios)    
	
	eof = .true.
	
	if (ios /= int(0)) then
		write(*,'(a)') ''	
		write(*,'(a)') ' converter error: no particle_data.dat file'
		write(*,'(a)') ''	
		write(*,'(a)') ' hit enter to exit'
		read *
		stop
	end if

	! set up pvd file
	open(2,file="p_data.pvd")
	write(2,'(a)') '<?xml version="1.0"?>'
	write(2,'(a)') '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">'
	write(2,'(a)') '<Collection>'
	
	read(1,*) np
	
	allocate(radius(np))
	allocate(position(np,3))
	allocate(velocity(np,3))
	allocate(alignment(np,3))
			
	read(1,*) radius
	    
    counter = 0
	
	do while ( eof ) 

		read(1,*,iostat=ios) time, & 
		          position(:,1), position(:,2), position(:,3), &
		          velocity(:,1), velocity(:,2), velocity(:,3), &
			  
		          alignment(:,1), alignment(:,2), alignment(:,3) 		          

		if ( ios < int(0) ) then
			eof = .false.			
		end if

		write(timestring,'(e15.8)') time 
		write(filestring,'(i15)') counter				
		write(pvdstring,'(a,a,a,a,a)') '<DataSet timestep="', &
									trim(adjustl(timestring)), &
									'" group="" part="0" file="p_data', &
									trim(adjustl(filestring)), &
									'.vtu"/>'
		write(2,'(a)') pvdstring
		
		write(vtustring,'(a,a,a)') 'p_data',trim(adjustl(filestring)),'.vtu' 					
        write(npstring,'(i15)') np
		
		open(3,file=vtustring)
						
        write(3,'(a)') '<?xml version="1.0"?>'
        write(3,'(a)') '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">'
        write(3,'(a)') '<UnstructuredGrid>'

        write(3,'(a,a,a)') '<Piece NumberOfPoints="',trim(adjustl(npstring)),'" NumberOfCells="0">'
        
        write(3,'(a)') '<Points>'        
        write(3,'(a)') '<DataArray type="Float32" Name="position" NumberOfComponents="3" format="ascii">'
        do i = 1,np
            write(3,'(3(1x,e15.8))') position(i,1),position(i,2),position(i,3)
        end do
        write(3,'(a)') '</DataArray>'        
        write(3,'(a)') '</Points>'
        
        write(3,'(a)') '<Cells>'
        write(3,'(a)') '<DataArray type="Int32" Name="connectivity" format="ascii">'
        write(3,'(a)') '0'
        write(3,'(a)') '</DataArray>'
        write(3,'(a)') '<DataArray type="Int32" Name="offsets" format="ascii">' 
        write(3,'(a)') '0'
        write(3,'(a)') '</DataArray>'
        write(3,'(a)') '<DataArray type="UInt8" Name="types" format="ascii">'
        write(3,'(a)') '1'
        write(3,'(a)') '</DataArray>'
        write(3,'(a)') '</Cells>'
        
        write(3,'(a)') '<PointData Vectors="vectors">'
        
        write(3,'(a)') '<DataArray type="Float32" Name="radius" NumberOfComponents="3" format="ascii">'
        do i = 1,np
            write(3,'(3(1x,e15.8))') radius(i),real(0.0,8),real(0.0,8)
        end do
        write(3,'(a)') '</DataArray>'        
        
        write(3,'(a)') '<DataArray type="Float32" Name="velocity" NumberOfComponents="3" format="ascii">'
        do i = 1,np
            write(3,'(3(1x,e15.8))') velocity(i,1),velocity(i,2),velocity(i,3)
        end do        
        write(3,'(a)') '</DataArray>'

	write(3,'(a)') '<DataArray type="Float32" Name="alignment" NumberOfComponents="3" format="ascii">'
        do i = 1,np
            write(3,'(3(1x,e15.8))') alignment(i,1),alignment(i,2),alignment(i,3)
        end do        
        write(3,'(a)') '</DataArray>'
        
        
        write(3,'(a)') '</PointData>'                        
        write(3,'(a)') '</Piece>'
        write(3,'(a)') '</UnstructuredGrid>'
        write(3,'(a)') '</VTKFile>'

        close(3)		
		counter = counter + 1
		
	end do

	write(2,'(a)') '</Collection>'
	write(2,'(a)') '</VTKFile>'

	close(1)
	close(2)
	



end program convert