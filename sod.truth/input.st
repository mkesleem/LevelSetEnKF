//Units: non-dimensional. This is just the Sod tube problem (simplified version of the problem on Page 129 of Arthur Rallu thesis)

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

  under GeometricEntities {

    under Plane[0] {
      Point_x = 0.59;
      Point_y = 0.0;
      Point_z = 0.0;
      Normal_x = 1.0;
      Normal_y = 0.0;
      Normal_z = 0.0;

      under InitialState {
        MaterialID = 0;
        Density = 0.125; 
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
    Density = 1.0; 
    VelocityX = 0.0;
    VelocityY = 0.0;
    VelocityZ = 0.0;
    Pressure = 1.0;
  }
  under Inlet2 {
    MaterialID = 0;
    Density = 0.125; 
    VelocityX = 0.0;
    VelocityY = 0.0;
    VelocityZ = 0.0;
    Pressure = 0.1;
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
  //MaxIts = 500;
  MaxTime = 0.2;
  CFL    = 0.55;
  //TimeStep = 0.00005;
  under Explicit {
    Type = RungeKutta2;
  }
}

under Output {
  Prefix = "results/";
  Solution = "solution";
  //Frequency = 1;
  TimeInterval = 0.002;
  Density = On;
  Velocity = On;
  Pressure = On;
  LevelSet0 = On;
  MaterialID = On;
  Temperature = On;
}
