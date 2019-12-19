!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      ! Fortran Version = 7
      subroutine check_version(version, warning)
              implicit none
              integer :: version, warning
!f2py         intent(in) :: version
!f2py         intent(out) :: warning
              if (version .NE. 7) then
                warning = 1
              else
                warning = 0
              end if
       end subroutine

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        subroutine simple_calculate_g2(numbers, rs, g_number, g_eta, cutoff, &
                                                       home, n, ridge)

              implicit none
              integer, dimension(n) :: numbers
              integer, dimension(1) :: g_number
              double precision, dimension(n, 3) :: rs
              double precision, dimension(3) :: home
              integer :: n
              double precision ::  g_eta, cutoff
              double precision :: ridge
!f2py         intent(in) :: numbers, rs, g_number
!f2py         intent(in) :: g_eta, cutoff, home
!f2py         intent(hide) :: n
!f2py         intent(out) :: ridge
              integer :: j, match, xyz
              double precision, dimension(3) :: Rij_
              double precision :: Rij, term

              ridge = 0.0d0
              do j = 1, n
                  match = compare(numbers(j), g_number(1))
                  if (match == 1) then
                    do xyz = 1, 3
                      Rij_(xyz) = rs(j, xyz) - home(xyz)
                    end do
                    Rij = sqrt(dot_product(Rij_, Rij_))
                !     term = exp(-g_eta*(Rij**2.0d0) / (cutoff ** 2.0d0))
                    term = exp(-1*exp(-2*g_eta)*(Rij**2) / (cutoff ** 2))
                    term = term * cutoff_fxn(Rij, cutoff)
                    ridge = ridge + term
                  end if
                end do

        CONTAINS

      function compare(try, val) result(match)
!     Returns 1 if try is the same set as val, 0 if not.
              implicit none
              integer, intent(in) :: try, val
              integer :: match
              if (try == val) then
                      match = 1
              else
                      match = 0
              end if
      end function compare

      function cutoff_fxn(r, cutoff)
              double precision :: r, cutoff, cutoff_fxn, pi
              if (r > cutoff) then
                      cutoff_fxn = 0.0d0
              else
                      pi = 4.0d0 * datan(1.0d0)
                      cutoff_fxn = 0.5d0 * (cos(pi*r/cutoff) + 1.0d0)
              end if

      end function

      end subroutine simple_calculate_g2



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

       subroutine calculate_g2(numbers, rs, g_numbers, g_etas, indices, cutoff, &
       c_index, home, atnum, ridges, forceT, virialT, m, n)

              implicit none
              integer, dimension(n) :: numbers, indices
              integer, dimension(m) :: g_numbers
              double precision, dimension(m) :: g_etas
              double precision, dimension(n, 3) :: rs
              double precision, dimension(3) :: home
              integer :: n ! number of neighbor atoms
              integer :: m ! length of fingerprint
              integer :: atnum ! atom number in the cell, atnum == max(indices)
              integer :: c_index
              double precision :: cutoff
              double precision :: eta
!f2py         intent(in) :: numbers, rs, g_numbers
!f2py         intent(in) :: g_etas, cutoff, home, indices, c_index, atnum
!f2py         intent(in, hide) :: n
!f2py         intent(in, hide) :: m
! xxxf2py         intent(hide) :: n
! xxxf2py         intent(hide) :: m
!f2py         intent(out) :: ridges, forceT, virialT
              double precision, dimension(m) :: ridges
              double precision, dimension(m, atnum, 3) :: forceT ! forces coefficient
              double precision, dimension(m, atnum, 6) :: virialT ! virial coefficient
              integer :: j, k, match, xyz
              double precision, dimension(3) :: Rij_, force_term
              double precision :: Rij, term, der_term

              ridges = 0.0d0
              forceT = 0.0d0
              virialT = 0.0d0
        !       do k = 1, m
        !         eta = g_etas(k)
                do j = 1, n
                    do xyz = 1, 3
                        Rij_(xyz) = rs(j, xyz) - home(xyz)
                    end do
                    Rij = sqrt(dot_product(Rij_, Rij_))
                    do k = 1, m
                        eta = g_etas(k)

                        match = compare(numbers(j), g_numbers(k))
                        if (match == 1) then

                        term = exp(-1*exp(-2*eta)*(Rij**2) / (cutoff ** 2))
                        !     term = term * cutoff_fxn(Rij, cutoff)
                        ridges(k) = ridges(k) + term * cutoff_fxn(Rij, cutoff)

                        ! WRITE(*,*) j,k,eta, Rij, cutoff
                        ! WRITE(*,*) term * cutoff_fxn(Rij, cutoff)

                        der_term = term * (der_cutoff_fxn(Rij, cutoff) - &
                        exp(-2*eta) * 2 * Rij * cutoff_fxn(Rij, cutoff)/(cutoff**2.0d0))

                        ! if (j == 1) then
                        !         WRITE(*,*) j,k,eta, Rij, cutoff
                        !         WRITE(*,*) der_term
                        ! end if
                        ! force
                        do xyz = 1, 3
                        ! delta_i R_{ij} = r_{ji} = r_i - r_j
                        ! force = -delta_i E along r_j - r_i
                                force_term(xyz) = der_term * Rij_(xyz)/Rij
                                forceT(k, indices(j)+1, xyz) = forceT(k, indices(j)+1, xyz) + force_term(xyz)
                                forceT(k, c_index+1, xyz) = forceT(k, c_index+1, xyz) + force_term(xyz)
                        end do

                        !     forceT(k, indices(j) + 1, 1) = forceT(k, indices(j) + 1, 1) + force_term(1)
                        !     forceT(k, indices(j) + 1, 2) = forceT(k, indices(j) + 1, 2) + force_term(2)
                        !     forceT(k, indices(j) + 1, 3) = forceT(k, indices(j) + 1, 3) + der_term3
                        !     forceT(k, c_index + 1, 1) = forceT(k, c_index + 1, 1) + force_term(1)
                        !     forceT(k, c_index + 1, 2) = forceT(k, c_index + 1, 2) + der_term2
                        !     forceT(k, c_index + 1, 3) = forceT(k, c_index + 1, 3) + der_term3
                        ! virial
                        ! voigt notation: https://en.wikipedia.org/wiki/Voigt_notation
                        virialT(k, indices(j)+1, 1) = virialT(k, indices(j)+1, 1) + force_term(1)*Rij_(1)
                        virialT(k, indices(j)+1, 2) = virialT(k, indices(j)+1, 2) + force_term(2)*Rij_(2)
                        virialT(k, indices(j)+1, 3) = virialT(k, indices(j)+1, 3) + force_term(3)*Rij_(3)
                        virialT(k, indices(j)+1, 4) = virialT(k, indices(j)+1, 4) + force_term(3)*Rij_(2)
                        virialT(k, indices(j)+1, 5) = virialT(k, indices(j)+1, 5) + force_term(3)*Rij_(1)
                        virialT(k, indices(j)+1, 6) = virialT(k, indices(j)+1, 6) + force_term(1)*Rij_(2)
                        virialT(k, c_index + 1, 1) = virialT(k, c_index+1, 1) + force_term(1)*Rij_(1)
                        virialT(k, c_index + 1, 2) = virialT(k, c_index+1, 2) + force_term(2)*Rij_(2)
                        virialT(k, c_index + 1, 3) = virialT(k, c_index+1, 3) + force_term(3)*Rij_(3)
                        virialT(k, c_index + 1, 4) = virialT(k, c_index+1, 4) + force_term(3)*Rij_(2)
                        virialT(k, c_index + 1, 5) = virialT(k, c_index+1, 5) + force_term(3)*Rij_(1)
                        virialT(k, c_index + 1, 6) = virialT(k, c_index+1, 6) + force_term(1)*Rij_(2)

                        ! virialT(k, indices(j)+1, 1) = virialT(k, indices(j)+1, 1) + force_term(1)*home(1)
                        ! virialT(k, indices(j)+1, 2) = virialT(k, indices(j)+1, 2) + force_term(2)*home(2)
                        ! virialT(k, indices(j)+1, 3) = virialT(k, indices(j)+1, 3) + force_term(3)*home(3)
                        ! virialT(k, indices(j)+1, 4) = virialT(k, indices(j)+1, 4) + force_term(3)*home(2)
                        ! virialT(k, indices(j)+1, 5) = virialT(k, indices(j)+1, 5) + force_term(3)*home(1)
                        ! virialT(k, indices(j)+1, 6) = virialT(k, indices(j)+1, 6) + force_term(1)*home(2)
                        ! virialT(k, c_index + 1, 1) = virialT(k, c_index+1, 1) + force_term(1)*home(1)
                        ! virialT(k, c_index + 1, 2) = virialT(k, c_index+1, 2) + force_term(2)*home(2)
                        ! virialT(k, c_index + 1, 3) = virialT(k, c_index+1, 3) + force_term(3)*home(3)
                        ! virialT(k, c_index + 1, 4) = virialT(k, c_index+1, 4) + force_term(3)*home(2)
                        ! virialT(k, c_index + 1, 5) = virialT(k, c_index+1, 5) + force_term(3)*home(1)
                        ! virialT(k, c_index + 1, 6) = virialT(k, c_index+1, 6) + force_term(1)*home(2)
                        end if
                    end do
                end do
                ! virialT = 0.5*virialT
        !       end do

      CONTAINS

      function compare(try, val) result(match)
!     Returns 1 if try is the same set as val, 0 if not.
              implicit none
              integer, intent(in) :: try, val
              integer :: match
              if (try == val) then
                      match = 1
              else
                      match = 0
              end if
      end function compare

      function cutoff_fxn(r, cutoff)
              double precision :: r, cutoff, cutoff_fxn, pi
              if (r > cutoff) then
                      cutoff_fxn = 0.0d0
              else
                      pi = 4.0d0 * datan(1.0d0)
                      cutoff_fxn = 0.5d0 * (cos(pi*r/cutoff) + 1.0d0)
              end if

      end function

      function der_cutoff_fxn(r, cutoff)
              double precision :: r, cutoff, der_cutoff_fxn, pi
              if (r > cutoff) then
                      der_cutoff_fxn = 0.0d0
              else
                      pi = 4.0d0 * datan(1.0d0)
                      der_cutoff_fxn = -0.5d0 * pi * sin(pi*r/cutoff) &
                       / cutoff
              end if
      end function


      end subroutine calculate_g2

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine calculate_g4(numbers, rs, g_numbers, g_zetas, &
                           g_etas, cutoff, indices, c_index, home, atnum, ridges, forceT, virialT, m, n)

              implicit none
              integer, dimension(n) :: numbers, indices
              integer, dimension(m, 2) :: g_numbers
              double precision, dimension(m) :: g_etas, g_zetas
              double precision, dimension(n, 3) :: rs
              double precision, dimension(3) :: home
              integer :: n ! number of neighbor atoms
              integer :: m ! length of fingerprint
              integer :: atnum ! atom number in the cell, atnum == max(indices)
              integer :: c_index
              double precision :: cutoff
              double precision :: eta, zeta
!f2py         intent(in) :: numbers, rs, g_numbers
!f2py         intent(in) :: g_etas, cutoff, home, indices, c_index, atnum
!f2py         intent(in, hide) :: n
!f2py         intent(in, hide) :: m
! xxxf2py         intent(hide) :: n
! xxxf2py         intent(hide) :: m
!f2py         intent(out) :: ridges, forceT, virialT
              double precision, dimension(m) :: ridges
              double precision, dimension(m, atnum, 3) :: forceT ! forces coefficient
              double precision, dimension(m, atnum, 6) :: virialT ! virial coefficient
              integer :: j, k, l, match, xyz
              double precision, dimension(3) :: Rij_, Rik_, Rjk_
              double precision :: Rij, Rik, Rjk, term, term_Rij, term_Rik, term_Rjk, term_cosi, term_cosj, term_cosk
              double precision :: der_Rij, der_i, der_j, der_k, der_cos, der_term1, der_term2, cosi, cosj, cosk

              ridges = 0.0d0
              forceT = 0.0d0
              virialT = 0.0d0
              do j = 1, n
                do k = (j + 1), n
                  do xyz = 1, 3
                    Rij_(xyz) = rs(j, xyz) - home(xyz)
                    Rik_(xyz) = rs(k, xyz) - home(xyz)
                    Rjk_(xyz) = rs(k, xyz) - rs(j, xyz)
                  end do
                  Rij = sqrt(dot_product(Rij_, Rij_))
                  Rik = sqrt(dot_product(Rik_, Rik_))
                  Rjk = sqrt(dot_product(Rjk_, Rjk_))

                  do l = 1, m
                    zeta = g_zetas(l)
                    eta = g_etas(l)

                    match = compare(numbers(j), numbers(k), g_numbers(l, 1),&
                              g_numbers(l, 2))
                    if (match == 1 .AND. Rjk <= cutoff) then

                        cosi = dot_product(Rij_, Rik_) / Rij / Rjk
                        term_cosi = (1.0d0 + cosi)**zeta
                        term_Rij = exp(-eta*(Rij**2/(cutoff ** 2.0d0)))*cutoff_fxn(Rij, cutoff)
                        term_Rik = exp(-eta*(Rik**2/(cutoff ** 2.0d0)))*cutoff_fxn(Rik, cutoff)
                        !       term_Rjk = exp(-eta*(Rjk**2/(cutoff ** 2.0d0)))*cutoff_fxn(Rjk, cutoff)
                        !       term = term_cosi*term_Rij*term_Rik*term_Rjk * 2.0d0**(1.0d0 - zeta)
                        term = cosi * term_Rij * term_Rik
                        ridges(l) = ridges(l) + term
                                ! ridges(l) = ridges(l) * 2.0d0**(1.0d0 - zeta)

                      ! force
                        !!!!!!! Behler G4 derivate
                        ! der_Rij = term_Rij * (der_cutoff_fxn(Rij, cutoff) - &
                        !         eta * 2 * Rij * cutoff_fxn(Rij, cutoff)/(cutoff**2.0d0))
                        ! der_cos = (1.0d0 + cosi)**(zeta-1) * zeta * (1/Rik - cosi/Rij)

                        ! der_term1 = term_cosi*term_Rik*term_Rjk * der_Rij * 2.0d0**(1.0d0 - zeta)
                        ! der_term2 = der_cos*term_Rij*term_Rik*term_Rjk * 2.0d0**(1.0d0 - zeta)
                        ! der_i = der_term1 + der_term2

                        ! cosj = -1 * dot_product(Rij_, Rjk_) / Rij / Rjk
                        ! term_cosj = (1.0d0 + cosj)**zeta
                        ! der_cos = (1.0d0 + cosj)**(zeta-1) * zeta * (1/Rjk - cosj/Rij)
                        ! der_term1 = term_cosj*term_Rik*term_Rjk * der_Rij * 2.0d0**(1.0d0 - zeta)
                        ! der_term2 = der_cos*term_Rij*term_Rik*term_Rjk * 2.0d0**(1.0d0 - zeta)
                        ! der_j = der_term1 + der_term2

                        ! cosk = dot_product(Rik_, Rjk_) / Rik / Rjk
                        ! term_cosk = (1.0d0 + cosk)**zeta
                        ! der_cos = -1 * (1.0d0 + cosk)**(zeta-1) * zeta * Rij / Rik / Rjk
                        ! der_term1 = term_cosk*term_Rik*term_Rjk * der_Rij * 2.0d0**(1.0d0 - zeta)
                        ! der_term2 = der_cos*term_Rij*term_Rik*term_Rjk * 2.0d0**(1.0d0 - zeta)
                        ! der_k = der_term1 + der_term2
                        !!!!!! End Behler G4 derivate

                        !!! AFS derivate
                        der_Rij = term_Rij * (der_cutoff_fxn(Rij, cutoff) - &
                                eta * 2 * Rij * cutoff_fxn(Rij, cutoff)/(cutoff**2.0d0))
                        der_cos = 1/Rik - cosi/Rij
                        der_term1 = cosi*term_Rik* der_Rij
                        der_term2 = der_cos*term_Rij*term_Rik
                        der_i = der_term1 + der_term2

                        cosj = -1 * dot_product(Rij_, Rjk_) / Rij / Rjk
                        der_cos = 1/Rjk - cosj/Rij
                        der_term1 = cosj*term_Rik*der_Rij
                        der_term2 = der_cos*term_Rij*term_Rik
                        der_j = der_term1 + der_term2

                        cosk = dot_product(Rik_, Rjk_) / Rik / Rjk
                        der_cos = -1 * Rij / Rik / Rjk
                        der_term1 = cosk*term_Rik*der_Rij
                        der_term2 = der_cos*term_Rij*term_Rik
                        der_k = der_term1 + der_term2
                        !!! End AFS derivate




                        do xyz = 1, 3
                      ! delta_i R_{ij} = r_{ji} = r_i - r_j
                      ! force = -delta_i E along r_j - r_i
                        !       force_term(xyz) = der_term * Rij_(xyz)/Rij
                              ! from atom i
                              forceT(l, c_index+1, xyz) = forceT(l, c_index+1, xyz) + der_i * Rij_(xyz)/Rij
                              ! from atom j
                              forceT(l, indices(j)+1, xyz) = forceT(l, indices(j)+1, xyz) + der_j * Rij_(xyz)/Rij
                              ! from atom k
                              forceT(l, indices(k)+1, xyz) = forceT(l, indices(k)+1, xyz) + der_k * Rij_(xyz)/Rij
                        end do

                      ! virial
                      ! from atom i
                        virialT(l, c_index + 1, 1) = virialT(l, c_index+1, 1) - der_i * Rij_(1)*Rij_(1)/Rij
                        virialT(l, c_index + 1, 2) = virialT(l, c_index+1, 2) - der_i * Rij_(2)*Rij_(2)/Rij
                        virialT(l, c_index + 1, 3) = virialT(l, c_index+1, 3) - der_i * Rij_(3)*Rij_(3)/Rij
                        virialT(l, c_index + 1, 4) = virialT(l, c_index+1, 4) - der_i * Rij_(3)*Rij_(2)/Rij
                        virialT(l, c_index + 1, 5) = virialT(l, c_index+1, 5) - der_i * Rij_(3)*Rij_(1)/Rij
                        virialT(l, c_index + 1, 6) = virialT(l, c_index+1, 6) - der_i * Rij_(1)*Rij_(2)/Rij
                        ! from atom j
                        virialT(l, indices(j)+1, 1) = virialT(l, indices(j)+1, 1) - der_j*Rij_(1)*Rij_(1)/Rij
                        virialT(l, indices(j)+1, 2) = virialT(l, indices(j)+1, 2) - der_j*Rij_(2)*Rij_(2)/Rij
                        virialT(l, indices(j)+1, 3) = virialT(l, indices(j)+1, 3) - der_j*Rij_(3)*Rij_(3)/Rij
                        virialT(l, indices(j)+1, 4) = virialT(l, indices(j)+1, 4) - der_j*Rij_(3)*Rij_(2)/Rij
                        virialT(l, indices(j)+1, 5) = virialT(l, indices(j)+1, 5) - der_j*Rij_(3)*Rij_(1)/Rij
                        virialT(l, indices(j)+1, 6) = virialT(l, indices(j)+1, 6) - der_j*Rij_(1)*Rij_(2)/Rij
                        ! from atom k
                        virialT(l, indices(k)+1, 1) = virialT(l, indices(k)+1, 1) - der_k*Rij_(1)*Rij_(1)/Rij
                        virialT(l, indices(k)+1, 2) = virialT(l, indices(k)+1, 2) - der_k*Rij_(2)*Rij_(2)/Rij
                        virialT(l, indices(k)+1, 3) = virialT(l, indices(k)+1, 3) - der_k*Rij_(3)*Rij_(3)/Rij
                        virialT(l, indices(k)+1, 4) = virialT(l, indices(k)+1, 4) - der_k*Rij_(3)*Rij_(2)/Rij
                        virialT(l, indices(k)+1, 5) = virialT(l, indices(k)+1, 5) - der_k*Rij_(3)*Rij_(1)/Rij
                        virialT(l, indices(k)+1, 6) = virialT(l, indices(k)+1, 6) - der_k*Rij_(1)*Rij_(2)/Rij

                    end if
                  end do
                end do
              end do


      CONTAINS

      function compare(try1, try2, val1, val2) result(match)
!     Returns 1 if (try1, try2) is the same set as (val1, val2), 0 if not.
              implicit none
              integer, intent(in) :: try1, try2, val1, val2
              integer :: match
              integer :: ntry1, ntry2, nval1, nval2
              ! First sort to avoid endless logical loops.
              if (try1 < try2) then
                      ntry1 = try1
                      ntry2 = try2
              else
                      ntry1 = try2
                      ntry2 = try1
              end if
              if (val1 < val2) then
                      nval1 = val1
                      nval2 = val2
              else
                      nval1 = val2
                      nval2 = val1
              end if
              if (ntry1 == nval1 .AND. ntry2 == nval2) then
                      match = 1
              else
                      match = 0
              end if

      end function compare

      function cutoff_fxn(r, cutoff)
              double precision :: r, cutoff, cutoff_fxn, pi
              if (r > cutoff) then
                      cutoff_fxn = 0.0d0
              else
                      pi = 4.0d0 * datan(1.0d0)
                      cutoff_fxn = 0.5d0 * (cos(pi*r/cutoff) + 1.0d0)
              end if

      end function

      function der_cutoff_fxn(r, cutoff)
              double precision :: r, cutoff, der_cutoff_fxn, pi
              if (r > cutoff) then
                      der_cutoff_fxn = 0.0d0
              else
                      pi = 4.0d0 * datan(1.0d0)
                      der_cutoff_fxn = -0.5d0 * pi * sin(pi*r/cutoff) &
                       / cutoff
              end if
      end function

      end subroutine calculate_g4

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
