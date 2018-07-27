prefactor = [-12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, ...
             0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]';
base = 6;
angles = prefactor * (pi / base);

orders = {'zyx', 'xzy', 'yxz', 'xyz', 'zxy', 'yzx', ...
          'zyz', 'zxz', 'yzy', 'yxy', 'xzx', 'xyx'}';
nmat = size(angles, 1)^3 * size(orders, 1);
ii = zeros(nmat, 1);
jj = zeros(nmat, 1);
kk = zeros(nmat, 1);
ang1 = zeros(nmat, 1);
ang2 = zeros(nmat, 1);
ang3 = zeros(nmat, 1);
order = strings([nmat, 1]);
mat_angle2dcm = zeros(nmat, 3, 3);

imat = 1;

for m = 1:size(orders)
    for i = 1:size(prefactor)
        for j = 1:size(prefactor)
            for k = 1:size(prefactor)

                ii(imat) = prefactor(i);
                jj(imat) = prefactor(j);
                kk(imat) = prefactor(k);

                ang1(imat) = angles(i);
                ang2(imat) = angles(j);
                ang3(imat) = angles(k);

                order(imat) = orders(m);

                mat_angle2dcm(imat, :, :) = angle2dcm(ang1(imat), ang2(imat), ang3(imat), ...
                                                      order(imat));
                imat = imat + 1;

            end
        end
    end
end

T_angle2dcm = table(ii, jj, kk, base * ones(size(ii)), ang1, ang2, ang3, order, ...
          mat_angle2dcm(:, 1, 1), mat_angle2dcm(:, 1, 2), mat_angle2dcm(:, 1, 3), ...
          mat_angle2dcm(:, 2, 1), mat_angle2dcm(:, 2, 2), mat_angle2dcm(:, 2, 3), ...
          mat_angle2dcm(:, 3, 1), mat_angle2dcm(:, 3, 2), mat_angle2dcm(:, 3, 3), ...
          'VariableNames', ...
          {'ii', 'jj', 'kk', 'base', 'ang0', 'ang1', 'ang2', 'order', ...
           'R00', 'R01', 'R02', 'R10', 'R11', 'R12', 'R20', 'R21', 'R22', ...
          }...
         );
head(T_angle2dcm)
writetable(T_angle2dcm, 'angle2dcm.csv');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

orders = {'zyx', 'xyz', 'zyz'}';
nmat = size(angles, 1)^3 * size(orders, 1);
ii = zeros(nmat, 1);
jj = zeros(nmat, 1);
kk = zeros(nmat, 1);
ang1 = zeros(nmat, 1);
ang2 = zeros(nmat, 1);
ang3 = zeros(nmat, 1);
order = strings([nmat, 1]);
mat_eul2rotm = zeros(nmat, 3, 3);

imat = 1;

for m = 1:size(orders)
    for i = 1:size(prefactor)
        for j = 1:size(prefactor)
            for k = 1:size(prefactor)

                ii(imat) = prefactor(i);
                jj(imat) = prefactor(j);
                kk(imat) = prefactor(k);

                ang1(imat) = angles(i);
                ang2(imat) = angles(j);
                ang3(imat) = angles(k);

                order(imat) = orders(m);

                mat_eul2rotm(imat, :, :) = eul2rotm([ang1(imat), ang2(imat), ang3(imat)], ...
                                                    char(order(imat)));
                imat = imat + 1;

            end
        end
    end
end

T_eul2rotm = table(ii, jj, kk, base * ones(size(ii)), ang1, ang2, ang3, order, ...
          mat_eul2rotm(:, 1, 1), mat_eul2rotm(:, 1, 2), mat_eul2rotm(:, 1, 3), ...
          mat_eul2rotm(:, 2, 1), mat_eul2rotm(:, 2, 2), mat_eul2rotm(:, 2, 3), ...
          mat_eul2rotm(:, 3, 1), mat_eul2rotm(:, 3, 2), mat_eul2rotm(:, 3, 3), ...
          'VariableNames', ...
          {'ii', 'jj', 'kk', 'base', 'ang0', 'ang1', 'ang2', 'order', ...
           'R00', 'R01', 'R02', 'R10', 'R11', 'R12', 'R20', 'R21', 'R22', ...
          }...
         );
head(T_eul2rotm)
writetable(T_eul2rotm, 'eul2rotm.csv');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nmat = size(angles, 1)^3;
ii = zeros(nmat, 1);
jj = zeros(nmat, 1);
kk = zeros(nmat, 1);
mat_eul2rotm = zeros(nmat, 3, 3);
q1_rotm = zeros(nmat, 1);
q2_rotm = zeros(nmat, 1);
q3_rotm = zeros(nmat, 1);
q4_rotm = zeros(nmat, 1);
q1_dcm = zeros(nmat, 1);
q2_dcm = zeros(nmat, 1);
q3_dcm = zeros(nmat, 1);
q4_dcm = zeros(nmat, 1);

imat = 1;

for m = 1:size(orders)
    for i = 1:size(prefactor)
        for j = 1:size(prefactor)
            for k = 1:size(prefactor)
                mat_eul2rotm(imat, :, :) = eul2rotm([angles(i), angles(j), angles(k)], 'ZYX');
                
                q_rotm = rotm2quat(reshape(mat_eul2rotm(imat, :, :), 3, 3));
                q1_rotm(imat) = q_rotm(1);
                q2_rotm(imat) = q_rotm(2);
                q3_rotm(imat) = q_rotm(3);
                q4_rotm(imat) = q_rotm(4);

                q_dcm = dcm2quat(reshape(mat_eul2rotm(imat, :, :), 3, 3));
                q1_dcm(imat) = q_dcm(1);
                q2_dcm(imat) = q_dcm(2);
                q3_dcm(imat) = q_dcm(3);
                q4_dcm(imat) = q_dcm(4);
                
                imat = imat + 1;
            end
        end
    end
end

T_rotm2quat = table(...
          mat_eul2rotm(:, 1, 1), mat_eul2rotm(:, 1, 2), mat_eul2rotm(:, 1, 3), ...
          mat_eul2rotm(:, 2, 1), mat_eul2rotm(:, 2, 2), mat_eul2rotm(:, 2, 3), ...
          mat_eul2rotm(:, 3, 1), mat_eul2rotm(:, 3, 2), mat_eul2rotm(:, 3, 3), ...
          q1_rotm, q2_rotm, q3_rotm, q4_rotm, ...
          q1_dcm, q2_dcm, q3_dcm, q4_dcm, ...          
          'VariableNames', ...
          {'R00', 'R01', 'R02', 'R10', 'R11', 'R12', 'R20', 'R21', 'R22', ...
           'q0_rotm', 'q1_rotm', 'q2_rotm', 'q3_rotm', ...
           'q0_dcm', 'q1_dcm', 'q2_dcm', 'q3_dcm', ...
          }...
         );
head(T_rotm2quat)
writetable(T_rotm2quat, 'rotm2quat.csv');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

angles = linspace(-2*pi, 2*pi, 33)';

vecs = [0, 0, 1;
        0, 1, 0;
        1, 0, 0;
        0, 1, 1;
        1, 0, 1;
        1, 1, 0;
        0, 0, -1;
        0, -1, 0;
        -1, 0, 0;
        0, -1, -1;
        -1, 0, -1;
        -1, -1, 0;
        0, 1, -1;
        1, 0, -1;
        1, -1, 0;
        0, -1, 1;
        -1, 0, 1;
        -1, 1, 0;
        ];


nmat = size(vecs, 1) * size(angles, 1);
ax1 = zeros(nmat, 1);
ax2 = zeros(nmat, 1);
ax3 = zeros(nmat, 1);
ang = zeros(nmat, 1);
mat_axang2rotm = zeros(nmat, 3, 3);
eul = zeros(nmat, 3);

imat = 1;

for i = 1:size(vecs, 1)
    for j = 1:size(angles)
        ax1(imat, 1) = vecs(i, 1);
        ax2(imat, 1) = vecs(i, 2);
        ax3(imat, 1) = vecs(i, 3);
        ang(imat, 1) = angles(j);
        mat_axang2rotm(imat, :, :) = axang2rotm([vecs(i, 1), vecs(i, 2), ...
                                                 vecs(i, 3), angles(j)]);
        eul(imat, :) = rotm2eul(reshape(mat_axang2rotm(imat, :, :), 3, 3));
        imat = imat + 1;
    end
end

T_axang2rotm = table(ax1, ax2, ax3, ang, ...
          mat_axang2rotm(:, 1, 1), mat_axang2rotm(:, 1, 2), mat_axang2rotm(:, 1, 3), ...
          mat_axang2rotm(:, 2, 1), mat_axang2rotm(:, 2, 2), mat_axang2rotm(:, 2, 3), ...
          mat_axang2rotm(:, 3, 1), mat_axang2rotm(:, 3, 2), mat_axang2rotm(:, 3, 3), ...
          eul(:, 1), eul(:, 2), eul(:, 3), ...
          'VariableNames', ...
          {'ax0', 'ax1', 'ax2', 'ang', ...
           'R00', 'R01', 'R02', 'R10', 'R11', 'R12', 'R20', 'R21', 'R22', ...
           'eul0', 'eul1', 'eul2', ...
          }...
         );
head(T_axang2rotm)
writetable(T_axang2rotm, 'axang2rotm.csv');
