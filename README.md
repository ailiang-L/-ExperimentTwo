# -ExperimentTwo
## max_loss 的讨论
### 当车车的最大通讯范围为200时的loss值
L_vv = L0vv + 10 * eta3* np.log10(d_vv /d0) +  X_eta4+zeta * Lcvv，其中  
X_eta4 : np.random.normal(0, 1.7)  
eta3: 1.68   
eta4: 1.7   
d0: 10   
Lcvv: 1.5   
L0vv: 85.0   
d_vv:200  
zeta:1，则L_vv的取值范围为多少？
当车辆的通信范围是200时，L_vv=85.0+21.857+X_eta4+1x1.5,且X_eta4均值为0，标准差为1.7的正太分布，99.7%的数据取值分布在[-5.1,+5.1]
那么车辆的最大loss值为85+21.857+5.1+1.5=113.575，即114

### 当无人机与无人机的最大范围即【0，100，0】，【500，100，500】的loss值
loss=95.45207540934052

### 当无人机与车的最大范围即车【0，0，0】，无人机【250，100，250】的loss值
loss=112.35227280032046

综上所述，最大的loss值为114

## min_loss的讨论
### 当车车的最小通讯范围为10时的loss值
L_vv = L0vv + 10 * eta3* np.log10(d_vv /d0) +  X_eta4+zeta * Lcvv，其中  
X_eta4 : np.random.normal(0, 1.7)  
eta3: 1.68   
eta4: 1.7   
d0: 10   
Lcvv: 1.5   
L0vv: 85.0   
d_vv:10  
zeta:1，则L_vv的取值范围为多少？
当车辆的通信范围是10时，L_vv=85.0+0+X_eta4+1x1.5,且X_eta4均值为0，标准差为1.7的正太分布，99.7%的数据取值分布在[-5.1,+5.1]
那么车辆的最小loss值为85+0-5.1+1.5=81.4，即81

### 当无人机与无人机的最小距离即【0，100，0】，【0，100，250】的loss值
loss=86.42117553942109

### 当无人机与车的小距离即车【0，0，0】，无人机【0，100，0】的loss值
loss=100.98703736269717

综上所述，最小的loss值为81

```js
    def ath_loss_V2V(self ,zeta_mode='reverse'):
        """
        计算城市环境中车辆之间的路径损耗。this model excerpted from <Path Loss Modeling for Vehicle-to-Vehicle Communications>
        :param target_position:
        :param zeta_mode: 路径损耗模式（'reverse', 'forward', 'convoy'）
        :return: 路径损耗
        """

        # 计算两车之间的距离
        d_vv = self.get_dis([0,0,0], [0,0,0])

        d_vv = d_vv if d_vv>=self.config['communication_config']['d0'] else self.config['communication_config']['d0']
        # 计算正态随机分布变量
        X_eta4 = np.random.normal(0, self.config['communication_config']["eta4"])
        print("d_vv:", d_vv, " x_eta4:", X_eta4)
        # 根据zeta_mode来决定zeta的值
        if zeta_mode == 'reverse':
            zeta = 1
        elif zeta_mode == 'forward':
            zeta = -1
        elif zeta_mode == 'convoy':
            zeta = 0
        else:
            raise ValueError("Invalid zeta_mode. Choose from 'reverse', 'forward', or 'convoy'.")
        print(self.config['communication_config']["L0vv"],"+",10 * self.config['communication_config'][
            "eta3"] * np.log10(d_vv / self.config['communication_config']["d0"]),"+" ,X_eta4,"+",zeta * self.config['communication_config']["Lcvv"])
        # 使用给定的公式计算路径损耗
        L_vv = self.config['communication_config']["L0vv"] + 10 * self.config['communication_config'][
            "eta3"] * np.log10(d_vv / self.config['communication_config']["d0"]) + \
               X_eta4 + zeta * self.config['communication_config']["Lcvv"]
        return L_vv
    def ath_loss_U2U(self):
        """
        :param target_position:
        :return: 路径损耗
        """
        v_c = 3 * 10 ** 8  # 光速
        d_uu = self.get_dis([0,100,0], [0,100,250])
        L_uu = 20 * np.log10(d_uu) + 20 * np.log10(self.config['communication_config']["fc"]) + 20 * np.log10(
            4 * np.pi / v_c)
        return L_uu
    def os_probability_U2V(self):
        """
        计算车辆与无人机之间的LoS概率。
        :param target_position:
        """
        uav_position = [0,100,0]
        vehicle_position = [0,0,0]
        d_vu = self.get_dis(uav_position, vehicle_position)

        # 计算仰角
        y_u = uav_position[1]
        elevation_angle = np.arcsin(y_u / d_vu)

        # 使用给定的公式计算LoS概率
        probability = 1 / (1 + self.config['communication_config']["eta1"] * np.exp(
            -self.config['communication_config']["eta2"] * (
                    elevation_angle - self.config['communication_config']["eta1"])))
        return probability

    def ath_loss_U2V(self):
        # 光速v_c，单位：m/s
        v_c = 3 * 10 ** 8

        # 计算LoS和NLoS的概率
        uav_position = [0,100,0]
        vehicle_position = [0,0,0]
        h_LoS = self.os_probability_U2V()
        h_NLoS = 1 - h_LoS
        d_vu = self.get_dis(uav_position, vehicle_position)
        # 计算自由空间路径损耗L^FS
        L_FS = 20 * np.log10(d_vu) + 20 * np.log10(self.config['communication_config']["fc"]) + 20 * np.log10(
            4 * np.pi / v_c)
        # 计算LoS和NLoS情况下的路径损耗
        L_LoS = L_FS + self.config['communication_config']["eta_LoS"]
        L_NLoS = L_FS + self.config['communication_config']["eta_NLoS"]
        # 计算总路径损耗
        L_total = h_LoS * L_LoS + h_NLoS * L_NLoS
        return L_total
```