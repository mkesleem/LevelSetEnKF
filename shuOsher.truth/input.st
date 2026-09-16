under Mesh {
  Type = ThreeDimensional;
  X0   = 0.0;
  Xmax = 1.0;
  Y0   = -0.05;
  Ymax = 0.05;
  Z0   = -0.05;
  Zmax = 0.05;
  NumberOfCellsX = 4000;
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
  UserDefinedStateCalculator = "IC/libUserDefinedState.so";
}

under BoundaryConditions {
  under Inlet {
    MaterialID = 0;
    Density = 3.85714300e+00; 
    VelocityX = 2.62936900e+00;
    VelocityY = 0.0;
    VelocityZ = 0.0;
    Pressure = 1.03333000e+01;
  }
  under Inlet2 {
    MaterialID = 0;
    Density = 8.00000000e-01; 
    VelocityX = 0.00000000e+00;
    VelocityY = 0.0;
    VelocityZ = 0.0;
    Pressure = 1.00000000e+00;
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
  MaxTime = 0.25;
  TimeStep = 1e-5;
  under Explicit {
    Type = RungeKutta2;
  }
}

under Output {
  Prefix = "results/";
  Solution = "solution";
  //Frequency = 1;
  TimeInterval = 0.0025;
  Density = On;
  Velocity = On;
  Pressure = On;
  LevelSet0 = On;
  MaterialID = On;
  Temperature = On;
}
