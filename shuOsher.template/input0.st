under Mesh {
  Type = ThreeDimensional;
  X0   = 0.0;
  Xmax = 1.0;
  Y0   = -0.05;
  Ymax = 0.05;
  Z0   = -0.05;
  Zmax = 0.05;
  NumberOfCellsX = 800;
  NumberOfCellsY = 1;
  NumberOfCellsZ = 1;

  BoundaryConditionX0   = Inlet;
  BoundaryConditionXmax = Inlet2;
  BoundaryConditionY0   = Symmetry;
  BoundaryConditionYmax = Symmetry;
  BoundaryConditionZ0   = Symmetry;
  BoundaryConditionZmax = Symmetry;
}

under Equations {
  under Material[0] {
    EquationOfState = StiffenedGas;
    under StiffenedGasModel {
      SpecificHeatRatio = 1.4;
      PressureConstant = 0.0;
    }
  }
  under Material[1] {
    EquationOfState = StiffenedGas;
    under StiffenedGasModel {
      SpecificHeatRatio = 1.4;
      PressureConstant = 0.0;
    }
  }
}

under InitialCondition {

  under GeometricEntities {

    under Plane[0] {
      Point_x = 0.1;
      Point_y = 0.0;
      Point_z = 0.0;
      Normal_x = 1.0;
      Normal_y = 0.0;
      Normal_z = 0.0;

      under InitialState {
        MaterialID = 0;
        Density = 1.0; 
        VelocityX = 0.0;
        VelocityY = 0.0;
        VelocityZ = 0.0;
        Pressure = 0.1;
      }
    }
  }

}

under BoundaryConditions {
  under Inlet {
    MaterialID = 0;
    Density = 3.857143; 
    VelocityX = 2.629369;
    VelocityY = 0.0;
    VelocityZ = 0.0;
    Pressure = 10.3333;
  }
  under Inlet2 {
    MaterialID = 0;
    Density = 1.0; 
    VelocityX = 0.0;
    VelocityY = 0.0;
    VelocityZ = 0.0;
    Pressure = 1.0;
  }
}

under Space {
  under NavierStokes {
    Flux = LocalLaxFriedrichs;
    under Reconstruction {
      Type = Linear;    
      VariableType = ConservativeCharacteristic; //not very sensitive
      SlopeNearInterface = NonZero;
      Limiter = VanAlbada; //GeneralizedMinMod;
      GeneralizedMinModCoefficient = 1.2;
    }
  }

  under LevelSet[0] {
    MaterialID = 1;
    under Reconstruction {
      Type = Linear;
      Limiter = None;
    }
  }

}

under MultiPhase {
  Flux = Numerical;
  ReconstructionAtInterface = Linear;  //slightly better than constant
  PhaseChange = RiemannSolution; //clearly better than extrapolation
}

under Time {
  Type = Explicit;
  MaxTime = 1.0e-10;
  TimeStep = 1.0e-10;
  under Explicit {
    Type = RungeKutta2;
  }
}

under Output {
  Prefix = "results/";
  Solution = "solution_0000";
  //Frequency = 1;
  TimeInterval = 0.0025;
  Density = On;
  Velocity = On;
  Pressure = On;
  LevelSet0 = On;
  MaterialID = On;
  Temperature = On;
}
